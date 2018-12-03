

import nengo
import nengo_spa as spa
import numpy as np

newCue = True
D = 64
s = spa.sym

# define trial structure =====
trial_duration = 2 #in_sec
cue_length     = 0.1
target_length  = 0.1
tar_ON         = 0.8 # delay period is 600 ms

class Timing(object):

    def __init__(self, trial_duration, cue_length, target_length, tar_ON):
        
        self.trial = 0 # trial counter
        self.cue_ind  = 0
        self.tar_ind = 0
        self.trial_duration = trial_duration
        self.cue_length  = cue_length
        self.target_length = target_length
        self.tar_ON   = tar_ON
        self.cues =  ['Cue_A','Cue_B','CUE_C','CUE_D']
        self.targets = ['VIS*RIGHT + AUD*LEFT','VIS*LEFT + AUD*RIGHT']

    def presentCues(self, t):
        if self.trial <= 100:
            # context 1
            cues = ['Cue_A','Cue_B']
        elif self.trial <= 200:
            # context 2
            cues = ['Cue_C','Cue_D']
        else:
            # randomized
            cues = ['Cue_A','Cue_B','CUE_C','CUE_D']

        if  (int(t) % self.trial_duration==0) and ((t-int(t)) == 0):
            self.trial = self.trial + 1
            self.cue_ind = np.random.randint(0,len(cues))

        if  (int(t)%self.trial_duration==0) and (0<= t-int(t) <= self.cue_length):
            index = self.cue_ind
            return cues[index]
        else:
            return '0'

    def presentTargets(self, t):
        targets = self.targets
        if  (int(t) % self.trial_duration==0) and ((t-int(t)) == 0):
            self.tar_ind = np.random.randint(0,len(targets))
            print("Trial = {0}, Cue = {1}, Target = {2}".format(self.trial, self.cues[self.cue_ind], self.targets[self.tar_ind]))
        if  (int(t)%self.trial_duration==0) and ( self.tar_ON <= t-int(t) <= (self.tar_ON + self.target_length)):
            index = self.tar_ind
            return targets[index]
        else:
            return '0'

# define vocabulary known in PFC
vocabPFC = spa.Vocabulary(D)
vocabPFC.populate('CUE_A; CUE_B; CUE_C; CUE_D; NOTHING')
vocabPFC.add('AUDITORY', vocabPFC.parse('CUE_A+CUE_C')) # go to visial
vocabPFC.add('VISUAL', vocabPFC.parse('CUE_B+CUE_D')) # go to sound
# define vocabulary known in MOTOR
vocabMOTOR = spa.Vocabulary(D)
vocabMOTOR.populate('RIGHT; LEFT; NOTHING')

with spa.Network(seed = 100) as model:

    exp = Timing( trial_duration, cue_length,target_length,tar_ON )

    # define population
    target  = spa.State(D, label='target')
    motor   = spa.State(D, feedback = 0.9 , label='motor')
    pfcCUE  = spa.State(D, feedback = 0.9, label='pfcCUE')
    pfcRULE = spa.State(D, feedback = 0.1, label='pfcRULE')
    md      = spa.State(D, feedback = 0, label='md')

    # connections
    conn = nengo.Connection(md.output, pfcRULE.input, synapse=0.05)
    conn.learning_rule_type = nengo.BCM(learning_rate = 1e-9 )

    # define inputs
    stim_cue = spa.Transcode( function = exp.presentCues, output_vocab = pfcCUE.vocab, label='stim Cue')
    stim_cue >> pfcCUE
    stim_target = spa.Transcode( function = exp.presentTargets, output_vocab = target.vocab, label='stim Target')
    stim_target >> target

    with spa.ActionSelection():
        spa.ifmax(0.3*spa.dot(s.NOTHING, pfcCUE),s.NOTHING >> pfcRULE)

        spa.ifmax(spa.dot(s.CUE_A + s.CUE_C, pfcCUE),
                        s.VIS >> md,
                        s.VIS >> pfcRULE)
        spa.ifmax(spa.dot(s.CUE_B + s.CUE_D, pfcCUE),
                        s.AUD >> md
                        )
        spa.ifmax(spa.dot(target, s.AUD*s.RIGHT+s.VIS*s.LEFT),
                        target*~pfcRULE >> motor)
