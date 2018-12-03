# PFC-MD circuit code version 4 --
# date: Saturday June 9 2018

import nengo
import nengo_spa as spa
import numpy as np
from nengo_extras.learning_rules import AML

## choose to run externally
runGUI = True
if runGUI == False:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import plot, draw, show

# define trial structure =====
trial_duration = 1 #in_sec
cue_length     = 0.1
target_length  = 0.1
tar_ON         = 0.5 # delay period is 600 ms

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

# define variables ====
dims  = 32
s = spa.sym
pfcFB = 0.05
motorFB = 1
ppcFB   = 1

# vocabmain = spa.Vocabulary(dims, rng=np.random.RandomState(1))
# vocabmain.populate('CUE_A; CUE_B; CUE_C; CUE_D; AUD; VIS; RIGHT; LEFT; RULE1; RULE2')

map = {
    'CUE_A':'CONTEXT1',
    'CUE_B':'CONTEXT1',
    'CUE_C':'CONTEXT2',
    'CUE_D':'CONTEXT2',
    }

# main code ==========
with spa.Network(seed = 10) as model:

    exp = Timing( trial_duration,cue_length,target_length,tar_ON )

    # define populations...
    cues = spa.State(dims, label = 'cues')
    targets = spa.State(dims, label =  'targets')
    motor = spa.State(dims, feedback = motorFB, label = 'motor')
    pfc  = spa.State(dims, feedback = pfcFB,  label = 'pfc')
    md  = spa.State(dims, feedback = 0.05, label = 'md')
    error  = spa.State(dims, feedback = 0.05, label = 'err')
    error2 = spa.State(dims, feedback = 0.05, label = 'err2')
    ppc  = spa.State(dims, feedback = ppcFB, label = 'ppc')

    # connect inputs to cues and targets
    stim_cues = spa.Transcode( function = exp.presentCues, output_vocab = cues.vocab)
    stim_cues >> cues

    stim_targets = spa.Transcode( function = exp.presentTargets, output_vocab = targets.vocab)
    stim_targets >> targets

    # vocab1 = targets.vocab
    # vocab1.add( 'Action1', vocab1.parse('VIS*RIGHT + AUD*LEFT') )
    # vocab1.add( 'Action2', vocab1.parse('VIS*LEFT + AUD*RIGHT') )

    vocab2 = ppc.vocab
    vocab2.add( 'CONTEXT1', vocab2.parse('CUE_A+CUE_B') )
    vocab2.add( 'CONTEXT2', vocab2.parse('CUE_C+CUE_D') )

    vocab = pfc.vocab
    vocab.add( 'RULE1', vocab.parse('CUE_A+CUE_C') )
    vocab.add( 'RULE2', vocab.parse('CUE_B+CUE_D') )


    # # ppc has cue history (perfect memory), assume direct sensory inputs
    # stim_cues >> ppc
    # # pfc gets inputs from cue responding "population"
    cues >> pfc
    stim_cues >> ppc

    with spa.ActionSelection():
          spa.ifmax( 0.5, s.X*0 >> pfc)
          spa.ifmax( 0.8*spa.dot(pfc, s.RULE1) +  0.8*spa.dot(targets, s.VIS*(s.RIGHT+s.LEFT)),
                       targets*~s.VIS >> motor )
          spa.ifmax( 0.8*spa.dot(pfc, s.RULE2) +  0.8*spa.dot(targets, s.AUD*(s.RIGHT+s.LEFT)),
                       targets*~s.AUD >> motor )

    # md = spa.WTAAssocMem(threshold = 0.5,mapping=map, input_vocab = pfc.vocab, label = 'MD')

    # learn pfc --> md computation
    md >> error
    -pfc >> error

    md_ens = list(md.all_ensembles)
    # pfc_ens = list(pfc.all_ensembles)
    error_ens = list(error.all_ensembles)
    error2_ens = list(error2.all_ensembles)

    for i, pfc_ens in enumerate(pfc.all_ensembles):
        c = nengo.Connection(pfc_ens, md_ens[i],
            transform = 0,
            learning_rule_type=nengo.PES(learning_rate=1e-4))
        nengo.Connection(error_ens[i], c.learning_rule)

    md >> pfc
    
    md >> error2 
    -ppc >> error2 
    
    for i, ppc_ens in enumerate(ppc.all_ensembles):
        c2 =nengo.Connection( ppc_ens, md_ens[i],
            transform = 0,
            learning_rule_type=nengo.PES(learning_rate=1e-4))
        nengo.Connection( error2_ens[i], c2.learning_rule)


    # pfc >> error2
    # -md >> error2


    # for i, md_e in enumerate(md.all_ensembles):
    #     print(i, md_e, pfc_ens[i])
    #     c2 = nengo.Connection(md_e, pfc_ens[i],
    #         function = lambda x: [0]*md_e.dimensions,
    #         learning_rule_type = nengo.PES(learning_rate = 1e-4) )
    #     nengo.Connection(error2_ens[i], c2.learning_rule)
