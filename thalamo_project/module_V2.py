
import nengo
import nengo_spa as spa
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show

model = spa.Network()
dims  = 32

s = spa.sym

pfcFB = 0.1

# define trial structure =====
trial_duration = 2 #in_sec
cue_length     = 0.2
target_length  = 0.2
tar_ON         = 0.8 # delay period is 600 ms

class Timing(object):

    def __init__(self, trial_duration, cue_length, target_length, tar_ON):
        self.rng = np.random.RandomState(400)
        self.trial = 0 # trial counter
        self.cue_ind  = 0
        self.tar_ind = 0
        self.trial_duration = trial_duration
        self.cue_length  = cue_length
        self.target_length = target_length
        self.tar_ON   = tar_ON

    def presentCues(self, t):
        if self.trial <= 100:
            # context 1
            cues = ['Cue_A','Cue_B']
        elif 101 <= self.trial <= 200:
            # context 2
            cues = ['Cue_C','Cue_D']
        else:
            # randomized
            cues = ['Cue_A','Cue_B','CUE_C','CUE_D']

        if  (int(t) % self.trial_duration==0) and ( (t-int(t)) == 0):
            self.trial = self.trial + 1
            self.cue_ind = np.random.randint(0,len(cues))
            print('Trial = ' + str(self.trial) )
        if  (int(t)%self.trial_duration==0) and (0<= t-int(t) <= self.cue_length):
            index = self.cue_ind
            return cues[index]
        else:
            return '0'

    def presentTargets(self, t):
        targets = ['VIS*RIGHT + AUD*LEFT','VIS*LEFT + AUD*RIGHT']
        if  (int(t) % self.trial_duration==0) and ( (t-int(t)) == 0):
            self.tar_ind = np.random.randint(0,len(targets))
        if  (int(t)%self.trial_duration==0) and ( self.tar_ON <= t-int(t) <= (self.tar_ON + self.target_length)):
            index = self.tar_ind
            return targets[index]
        else:
            return '0'

with model:
    # exp = Timing( trial_duration,cue_length,target_length,tar_ON )


    # define populations...
    cues = spa.State(dims, label = 'cues')
    targets = spa.State(dims, label =  'targets')
    motor = spa.State(dims, feedback = 1, label = 'motor')
    pfc  = spa.State(dims, feedback = pfcFB,  label = 'pfc')
    contextEncoder = spa.State(dims, feedback = 1, label = 'contextEncoder')

    def cuesFN(t):
        if 0.1<t<0.2:
            return 'CUE_A'
        else:
            return '0'

    stim_sensory1 = spa.Transcode( cuesFN, output_vocab = cues.vocab)
    stim_sensory1 >> cues

    def targetFN(t):
        if 0.8<t<0.9:
            return 'AUD*RIGHT+VIS*LEFT'
        else:
            return '0'

    stim_sensory2 = spa.Transcode( targetFN, output_vocab = targets.vocab)
    stim_sensory2 >> targets

    vocab = pfc.vocab
    vocab.add('RULE1', vocab.parse('CUE_A+CUE_C'))
    vocab.add('RULE2', vocab.parse('CUE_B+CUE_D'))
    # vocab.add('CONTEXT1', vocab.parse('CUE_A+CUE_B'))
    # vocab.add('CONTEXT2', vocab.parse('CUE_C+CUE_D'))

    vocab2 = contextEncoder.vocab
    vocab2.add('CONTEXT1', vocab2.parse('CUE_A+CUE_B'))
    vocab2.add('CONTEXT2', vocab2.parse('CUE_C+CUE_D'))

    cues >> pfc
    cues >> contextEncoder

    with spa.ActionSelection():
        spa.ifmax( 0.5, s.X*0 >> pfc)
        spa.ifmax( 0.8*spa.dot(pfc, s.RULE1) +  0.8*spa.dot(targets, s.VIS*(s.RIGHT+s.LEFT)),
                       targets*~s.VIS >> motor )
        spa.ifmax( 0.8*spa.dot(pfc, s.RULE2) +  0.8*spa.dot(targets, s.AUD*(s.RIGHT+s.LEFT)),
                       targets*~s.AUD >> motor )

    md = spa.ThresholdingAssocMem(threshold = 0.5, input_vocab = pfc.vocab, label = 'MD')
    pfc >> md
    md*0.8 >> pfc
