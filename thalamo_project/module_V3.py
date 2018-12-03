
import nengo
import nengo_spa as spa
import numpy as np
from nengo_extras.learning_rules import AML

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
######################################################################
dims  = 32
s = spa.sym

pfcFB = 0.05

###############################################################################
# cheating approach....
map = {
    'CUE_A': 'CONTEXT1',
    'CUE_B': 'CONTEXT1',
    'CUE_C': 'CONTEXT2',
    'CUE_D': 'CONTEXT2',
    }
###############################################################################


vocabmain = spa.Vocabulary(32, rng=np.random.RandomState(1))
vocabmain.populate('CUE_A; CUE_B; CUE_C; CUE_D; AUD; VIS; RIGHT; LEFT; RULE1; RULE2')

#############

trial_duration = 2 #in_sec
cue_length     = 0.2
target_length  = 0.2
tar_ON         = 0.8 # delay period is 600 ms

class Experiment (object):
    
    def __init__(self, trial_duration, cue_length, target_length, tar_ON):
        self.trial_counter = 0
        self.trial_duration = trial_duration
        self.cue_length  = cue_length
        self.target_length = target_length
        self.tar_ON   = tar_ON
    
    def presentCues(self, t):
        if  (int(t)%self.trial_duration==0)  and (0<= t-int(t) <= self.cue_length):
            self.trial_counter = self.trial_counter + 1
            return 'Cue_A'
        else:
            return '0'

    def presentTargets(self, t):
        if  (int(t)%self.trial_duration==0)  and ( self.tar_ON <= t-int(t) <= (self.tar_ON + self.target_length)):
            self.trial_counter = self.trial_counter + 1
            return 'AUD*RIGHT + VIS*RIGHT'
        else:
            return '0'

with spa.Network() as model:
    
    exp = Experiment( trial_duration,cue_length,target_length,tar_ON )

    # define populations...
    cues = spa.State(dims, label = 'cues')
    targets = spa.State(dims, label =  'targets')
    motor = spa.State(dims, feedback = 1, label = 'motor')
    pfc  = spa.State(dims, feedback = pfcFB,  label = 'pfc')
    ppc  = spa.State(dims, feedback = 1, label = 'ppc')

    # connect inputs to cues and targets
    stim_sensory1 = spa.Transcode( exp.presentCues, output_vocab = cues.vocab)
    stim_sensory1 >> cues

    stim_sensory2 = spa.Transcode( exp.presentTargets, output_vocab = targets.vocab)
    stim_sensory2 >> targets

    vocab = pfc.vocab
    vocab.add( 'RULE1', vocab.parse('CUE_A+CUE_C') )
    vocab.add( 'RULE2', vocab.parse('CUE_B+CUE_D') )
    # vocab.add('CONTEXT1', vocab.parse('CUE_A+CUE_B'))
    # vocab.add('CONTEXT2', vocab.parse('CUE_C+CUE_D'))

    cues >> pfc
    stim_sensory1 >> ppc

    with spa.ActionSelection():
        spa.ifmax( 0.5, s.X*0 >> pfc)
        spa.ifmax( 0.8*spa.dot(pfc, s.RULE1) +  0.8*spa.dot(targets, s.VIS*(s.RIGHT+s.LEFT)),
                       targets*~s.VIS >> motor )
        spa.ifmax( 0.8*spa.dot(pfc, s.RULE2) +  0.8*spa.dot(targets, s.AUD*(s.RIGHT+s.LEFT)),
                       targets*~s.AUD >> motor )

    md = spa.WTAAssocMem(threshold = 0.5,mapping=map, input_vocab = pfc.vocab, label = 'MD')
    pfc >> md
    md*0.8 >> pfc

    # connA = nengo.Connection(md.output, pfc.input,
    # solver = nengo.solvers.LstsqL2(weights=True))
    # connA.learning_rule_type = nengo.BCM(learning_rate=5e-10)
