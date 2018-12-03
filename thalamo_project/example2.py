import nengo
import nengo_spa as spa
import numpy as np

# define trial structure =====
trial_duration = 1 #in_sec
cue_length     = 0.2
target_length  = 0.2
tar_ON         = 0.6 # delay period is 600 ms

class Timing(object):

    def __init__(self, trial_duration, cue_length, target_length, tar_ON):
        self.trial = 0 # trial counter
        self.cue_ind  = 0
        self.tar_ind = 0
        self.trial_duration = trial_duration
        self.cue_length  = cue_length
        self.target_length = target_length
        self.tar_ON   = tar_ON
        self.cues =  ['CUE_A','CUE_B','CUE_C','CUE_D']
        self.targets = ['VIS*RIGHT + AUD*LEFT','VIS*LEFT + AUD*RIGHT']

    def presentCues(self, t):
        if self.trial <= 100:
            # context 1
            cues = ['CUE_A','CUE_B']
        elif 101 <= self.trial <= 200:
            # context 2
            cues = ['CUE_C','CUE_D']
        else:
            # randomized
            cues = ['CUE_A','CUE_B','CUE_C','CUE_D']

        if  (int(t) % self.trial_duration==0) and ( (t-int(t)) == 0):
            self.trial = self.trial + 1
            self.cue_ind = np.random.randint(0,len(cues))

        if  (int(t)%self.trial_duration==0) and (0<= t-int(t) <= self.cue_length):
            index = self.cue_ind
            return cues[index]
        else:
            return '0'

    def presentTargets(self, t):
        targets = self.targets
        if  (int(t) % self.trial_duration==0) and ( (t-int(t)) == 0):
            self.tar_ind = np.random.randint(0,len(targets))
            print("Trial = {0}, Cue = {1}, Target = {2}".format(self.trial, self.cues[self.cue_ind], self.targets[self.tar_ind]))
        if  (int(t)%self.trial_duration==0) and ( self.tar_ON <= t-int(t) <= (self.tar_ON + self.target_length)):
            index = self.tar_ind
            return targets[index]
        else:
            return '0'

class Environment(object):

    def __init__(self, size=20, dt=0.001):

        self.size = size
        self.x = size / 2.
        self.y = 9.
        self.th = 0

        self.dt = dt

        self.svg_open = '<svg width="100%%" height="100%%" viewbox="0 0 {0} {1}">'.format(self.size, self.size)
        self.walls    = '<rect width="{0}" height="{1}" style="fill:white;stroke:black;stroke-width:1"/>'.format(self.size, self.size)

        self.target1 = '''
        <rect width = "1.5" height = "1.5" style="fill:{0}" transform="translate({1},{2}) rotate({3})"/>
        '''
        self.target2 = '''
        <rect width = "1.5" height = "1.5" style="fill:{0}" transform="translate({1},{2}) rotate({3})"/>
        '''
        self.cue = '''
        <rect width = "1.5" height = "1.5" style="fill:{0}" transform="translate({1},{2}) rotate({3})"/>
        '''

        self.agent_template = '''
        <polygon points="0.25,0.25 -0.25,0.25 0,-0.5" style="fill:blue" transform="translate({0},{1}) rotate({2})"/>
        '''
        self.svg_close = '</svg>'

        self._nengo_html_ = ''

    def __call__(self, t, v):

        self.th += v[1] * self.dt
        self.x += np.cos(self.th)*v[0]*self.dt
        self.y += np.sin(self.th)*v[0]*self.dt

        self.x = np.clip(self.x, 0, self.size)
        self.y = np.clip(self.y, 0, self.size)
        if self.th > np.pi:
            self.th -= 2*np.pi
        if self.th < -np.pi:
            self.th += 2*np.pi

        direction = self.th * 180. / np.pi + 90.

        self._nengo_html_ = self.svg_open
        self._nengo_html_ += self.walls
        self._nengo_html_ += self.agent_template.format(self.x, self.y, 0)

        def colormapping(v):
         if np.dot(vocab.parse('CUE_A').v, v) >= 0.6:
             return 'cyan'
         elif np.dot(vocab.parse('CUE_B').v, v) >= 0.6:
             return 'magenta'
         elif spa.dot(vocab.parse('CUE_C').v, v) >= 0.6:
             return 'cyan'
         elif np.dot(vocab.parse('CUE_D').v, v) >= 0.6:
             return 'magenta'
         else:
             return 'black'
             
        def colormapping_Targets(v):
         if np.dot(vocab.parse('AUD*RIGHT+VIS*LEFT').v, v) >= 0.6:
             return ['red','green']
         elif np.dot(vocab.parse('AUD*LEFT+VIS*RIGHT').v, v) >= 0.6:
             return ['green','red']
         else:
             return ['black','black']

        # clr = colormapping(v[2:16])
        clr = 'black'
        clr2 = colormapping(v[2:])
        # print(clr)

        self._nengo_html_ += self.target1.format(clr2[1], 1, 1, 0)
        self._nengo_html_ += self.target2.format(clr2[0], 7.5, 1, 0)
        self._nengo_html_ += self.cue.format(clr, 4.5, 4.5, 0)
        self._nengo_html_ += self.svg_close

        return self.x, self.y, self.th

# Dimensions of the SPA vocabulary
DIM = 16 #32

vocab   = spa.Vocabulary(dimensions=DIM)
vocab.populate('CUE_A; CUE_B; CUE_C; CUE_D')

with spa.Network(seed=13) as model:
    
    exp = Timing( trial_duration, cue_length,target_length,tar_ON )

    input = spa.State(vocab, subdimensions = DIM, represent_identity=False, label = 'input' )
    input2 = spa.State(vocab, subdimensions = DIM, represent_identity=False, label = 'input2' )

    # Define the vocabulary of semantic pointers that will be used

    env = nengo.Node(
        Environment(
            size = 10,
        ),
        size_in = 2+16,
        size_out = 3,
    )

    velocity_input = nengo.Node([0, 0])

    nengo.Connection(velocity_input, env[:2])

    # stim_cue    = spa.Transcode( function = exp.presentCues, output_vocab = vocab, label='stim Cue')
    # stim_cue >> input
    # nengo.Connection( input.all_ensembles[0],  env[2:16],  synapse = None )
    
    stim_target    = spa.Transcode( function = exp.presentTargets, output_vocab = vocab, label='stim Cue')
    stim_target >> input2
    nengo.Connection( input2.all_ensembles[0], env[2:],  synapse = None )

    # nengo.Connection( stim_node, env )
