
'''
  thalamus latest 
'''

import nengo
import nengo_spa as spa
import numpy as np

newCue = True
D = 64
s = spa.sym

# define trial structure =====
trial_duration = 1 #in_sec
cue_length     = 0.2
target_length  = 0.2
tar_ON         = 0.4 # delay period is 600 ms

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
    
    def presentCuesScalar(self, t):
        if self.trial <= 100:
            # context 1
            cues = ['CUE_A','CUE_B']
            cues_map = [-1,1]
        elif 101 <= self.trial <= 200:
            # context 2
            cues = ['CUE_C','CUE_D']
            cues_map = [-2,2]
        else:
            # randomized
            cues = ['CUE_A','CUE_B','CUE_C','CUE_D']
            cues_map = [-1,1,-2,2]

        
        if  (int(t)%self.trial_duration==0) and (0<= t-int(t) <= self.cue_length):
            index = self.cue_ind
            return cues_map[index]
        else:
            return 0
        

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
            
    def presentTargetScalar(self, t):
        targets_map = [1,-1]
        if  (int(t)%self.trial_duration==0) and (0<= t-int(t) <= self.cue_length):
            index = self.tar_ind
            return targets_map[index]
        else:
            return 0

class Environment(object):

    def __init__(self, size=20, dt=0.001):
        self.Accum =  []
        self.trial_duration = 1 #in_sec
        self.cue_length     = 0.2
        self.target_length  = 0.2
        self.tar_ON         = 0.4 # delay period is 600 ms

        self.size = size
        self.x = size / 2.
        self.y = 8.
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
        <rect width = "0.5" height = "1" style="fill:blue" transform="translate({0},{1}) rotate({2})"/>
        '''
        self.svg_close = '</svg>'

        self._nengo_html_ = ''

    def __call__(self, t, v):
        
        if len(self.Accum) > 0:
            mu = np.mean(self.Accum)
            std  = np.std(self.Accum)
            buffer_zscore_ = (self.Accum - mu)/std
            if v[65] > 1.5*buffer_zscore_ :
                print('OK pos')
                angle = 45
            elif v[65] < -1.5*buffer_zscore_ :
                print('OK neg')
                angle = 45
            else:
                angle = 0
        else:
            angle = 45
        self.Accum.append(v[65])
            
        if (int(t)%self.trial_duration==0) and ( (self.tar_ON) <= t-int(t) <= (self.tar_ON + self.target_length + 0.1) ):
            self.th += angle*45*self.dt
            self.x  += np.cos(self.th)*0*self.dt
            self.y  += np.sin(self.th)*0*self.dt
        else:
            self.th = 0*self.dt
            self.x  += np.cos(self.th)*0*self.dt
            self.y  += np.sin(self.th)*0*self.dt

        # self.x = np.clip(self.x, 0, self.size)
        # self.y = np.clip(self.y, 0, self.size)
        if self.th > np.pi:
            self.th -= 2*np.pi
        if self.th < -np.pi:
            self.th += 2*np.pi

        direction = self.th * 180. / np.pi 

        self._nengo_html_ = self.svg_open
        self._nengo_html_ += self.walls
        self._nengo_html_ += self.agent_template.format(self.x, self.y, direction)

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
         if np.dot(vocab.parse('AUD*LEFT').v, v) >= 0.5:
             return ['red','green']
         elif np.dot(vocab.parse('AUD*RIGHT').v, v) >= 0.5:
             return ['green','red']
         else:
             return ['black','black']

        clr = colormapping(v[:64])
      
        clr2 = colormapping_Targets(v[64:64*2])

        self._nengo_html_ += self.target1.format(clr2[1], 1, 1, 0)
        self._nengo_html_ += self.target2.format(clr2[0], 7.5, 1, 0)
        self._nengo_html_ += self.cue.format(clr, 4.5, 4.5, 0)
        self._nengo_html_ += self.svg_close

        return self.x, self.y, self.th



# vocabulary
vocab = spa.Vocabulary(D)
vocab.populate('CUE_A; CUE_B; CUE_C; CUE_D; RIGHT; LEFT; VIS; AUD; NOTHING')
moves       = ['RIGHT','LEFT']
movesMap    = {k : k for k in moves}
assoRule    = { 'CUE_A': 'VIS',
                'CUE_C': 'VIS',
                'CUE_B': 'AUD',
                'CUE_D': 'AUD'}

with spa.Network(seed = 900) as model:
    # Running trial
    exp = Timing( trial_duration, cue_length,target_length,tar_ON )

    # define population
    pfcCUE      = spa.State(vocab, feedback=0.9, label='pfcCUE')
    pfcRULEmemo = spa.State(vocab, feedback=0.3, label='pfcRULEmemo',feedback_synapse=0.25)
    mdCUE       = spa.State(vocab, label='mdCUE')
    mdRULE      = spa.State(vocab, label='mdRULE')
    ppc         = spa.State(vocab, feedback = 0.9, label = 'ppc',feedback_synapse=0.25)
    errorPPC    = spa.State(vocab, feedback = 0.05, label = 'errPPC')

    # define inputs
    stim_cue    = spa.Transcode( function = exp.presentCues, output_vocab = vocab, label='stim Cue')
    stim_target = spa.Transcode( function = exp.presentTargets, output_vocab = vocab, label='stim Target')

    # associative memory CUE -> RULE
    pfcRULEasso = spa.ThresholdingAssocMem(0.3, input_vocab=vocab,
                                        mapping=assoRule,
                                        function=lambda x: x > 0,
                                        label='pfcRULEasso')
    stim_cue >> pfcRULEasso
    pfcRULEasso >> pfcRULEmemo

    #####   toDo: TRANSFORM THE pfcRULE into a Winner-take-all network #####

    #cleanup = spa.WTAAssocMem(0.3, vocab, function=lambda x: x > 0)

    #####           ####

    # connections
    stim_cue >> ppc
    stim_cue >> pfcCUE

    motorClean  = spa.WTAAssocMem(threshold=0.1, input_vocab=vocab,
                                        mapping = movesMap,
                                        function=lambda x: 1 if x > 0 else 0,
                                        label = 'motorClean')

    nengo.Connection(mdRULE.output, pfcRULEmemo.input, transform=5,synapse=0.05)


    with spa.ActionSelection():
        #spa.ifmax(0.3*spa.dot(s.NOTHING, pfcCUE),s.NOTHING >> pfcRULEmemo)

        #spa.ifmax(spa.dot(s.CUE_A + s.CUE_C, pfcCUE),
        #                s.VIS >> pfcRULE)
        #spa.ifmax(spa.dot(s.CUE_B + s.CUE_D, pfcCUE),
        #                s.AUD >> pfcRULE)
        spa.ifmax(spa.dot(pfcRULEasso, s.AUD), s.AUD >> mdRULE)
        spa.ifmax(spa.dot(pfcRULEasso, s.VIS), s.VIS >> mdRULE)
        spa.ifmax(spa.dot(stim_target, s.AUD*s.RIGHT+s.VIS*s.LEFT),
                          stim_target*~pfcRULEmemo >> motorClean)
        spa.ifmax(spa.dot(stim_target, s.VIS*s.RIGHT+s.AUD*s.LEFT),
                          stim_target*~pfcRULEmemo >> motorClean)

    # learning

    mdCUE_ens       = list(mdCUE.all_ensembles)
    mdRULE_ens      = list(mdRULE.all_ensembles)
    ppc_ens         = list(ppc.all_ensembles)
    errorPPC_ens    = list(errorPPC.all_ensembles)

    -ppc    >> errorPPC
    mdCUE   >> errorPPC
    for i, ppc_e in enumerate(ppc_ens):
        c2 = nengo.Connection(ppc_e, mdCUE_ens[i],
            #function = lambda x: [0]*md_e.dimensions,
            transform = 0,
            learning_rule_type = nengo.PES(learning_rate = 1e-4) )
        nengo.Connection(errorPPC_ens[i], c2.learning_rule)

    # damage md
    damage = nengo.Node(0, label='damage')
    for i, mdE_e in enumerate(mdRULE_ens):
        nengo.Connection(damage, mdE_e.neurons,
                        transform=10*np.ones((mdE_e.n_neurons,1)),
                        synapse=None)

    for i, mdC_e in enumerate(mdCUE_ens):
        nengo.Connection(damage, mdC_e.neurons,
                        transform=10*np.ones((mdC_e.n_neurons,1)),
                        synapse=None)
    
    #### connect to display
    env = nengo.Node(
        Environment(
            size = 10,
        ),
        size_in = 64+64+1,
        size_out = 3,
    )
    
    nengo.Connection( stim_cue.input, env[:64],  synapse = None )
    nengo.Connection( stim_target.input, env[64:64*2],  synapse = None )
    
    def mapdirection_to_scalar(x):
        # returns values in degrees.
        if np.dot( vocab.parse('LEFT').v, x ) >= 0.2:
            return 45
        elif np.dot(vocab.parse('RIGHT').v, x) >= 0.2:
            return -45
        else:
            return 0
            
    # ens = nengo.Ensemble(500, dimensions = 64)
    # nengo.Connection(motorClean.output, ens, synapse=None)
    input3 = spa.State( vocab, subdimensions = 64, represent_identity = False)
    
    turn_direction = nengo.Ensemble(n_neurons=200, dimensions=1)

    motorClean >> input3
    nengo.Connection( input3.all_ensembles[0],turn_direction, function = mapdirection_to_scalar )
    nengo.Connection( turn_direction, env[65], synapse  = None)
    
    
    
