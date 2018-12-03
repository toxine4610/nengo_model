

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

# vocabulary
vocab = spa.Vocabulary(D)
vocab.populate('CUE_A; CUE_B; CUE_C; CUE_D; RIGHT; LEFT; VIS; AUD; NOTHING')
moves       = ['RIGHT','LEFT']
movesMap    = {k : k for k in moves}

with spa.Network(seed = 100) as model:
    # Running trial
    exp = Timing( trial_duration, cue_length,target_length,tar_ON )
    
    # define population
    pfcCUE      = spa.State(vocab, feedback=0.9, label='pfcCUE')
    pfcRULE     = spa.State(vocab, feedback=0.3, label='pfcRULE',feedback_synapse=0.25)
    md          = spa.State(vocab, label='md')
    ppc         = spa.State(vocab, feedback = 0.9, label = 'ppc',feedback_synapse=0.25)
    errorPPC    = spa.State(vocab, feedback = 0.05, label = 'errPPC')
    errorRULE   = spa.State(vocab, feedback = 0.05, label = 'errRULE')
    
    # define inputs
    stim_cue    = spa.Transcode( function = exp.presentCues, output_vocab = vocab, label='stim Cue')
    stim_target = spa.Transcode( function = exp.presentTargets, output_vocab = vocab, label='stim Target')
    
    # connections
    nengo.Connection(md.output, pfcRULE.input, synapse=0.05)
    #md >> pfcRULE
    stim_cue >> ppc
    stim_cue >> pfcCUE
    
    
    motorClean  = spa.WTAAssocMem(threshold=0.1, input_vocab=vocab,
                                        mapping = movesMap,
                                        label = 'motorClean')
    #ppc     >> ppcClean
    #motor   >> motorClean
    
    pfcRULE - md >> pfcRULE 
    
    with spa.ActionSelection():
        spa.ifmax(0.3*spa.dot(s.NOTHING, pfcCUE),s.NOTHING >> pfcRULE)

        spa.ifmax(spa.dot(s.CUE_A + s.CUE_C, pfcCUE),
                        s.VIS >> pfcRULE)
        spa.ifmax(spa.dot(s.CUE_B + s.CUE_D, pfcCUE),
                        s.AUD >> pfcRULE)
        spa.ifmax(spa.dot(stim_target, s.AUD*s.RIGHT+s.VIS*s.LEFT),
                          stim_target*~pfcRULE >> motorClean)
        spa.ifmax(spa.dot(stim_target, s.VIS*s.RIGHT+s.AUD*s.LEFT),
                          stim_target*~pfcRULE >> motorClean)
    
    # learning
    #md       >> errorRULE
    #-pfcRULE >> errorRULE
    
    md_ens          = list(md.all_ensembles)
    ppc_ens         = list(ppc.all_ensembles)
    pfcRULE_ens     = list(pfcRULE.all_ensembles)
    errorPPC_ens    = list(errorPPC.all_ensembles)
    errorRULE_ens   = list(errorRULE.all_ensembles)

    #for i, ensamble in enumerate(pfcRULE_ens):
    #    c = nengo.Connection(ensamble, md_ens[i],
    #                        transform = 0,
    #                        learning_rule_type=nengo.PES(learning_rate=1e-4))
    #    nengo.Connection(errorRULE_ens[i], c.learning_rule)
        
    -ppc >> errorPPC
    md >> errorPPC
    
    
    for i, md_e in enumerate(ppc_ens):
        c2 = nengo.Connection(md_e, md_ens[i],
            #function = lambda x: [0]*md_e.dimensions,
            transform = 0,
            learning_rule_type = nengo.PES(learning_rate = 1e-4) )
        nengo.Connection(errorPPC_ens[i], c2.learning_rule)

    
