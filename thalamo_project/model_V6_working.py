

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
assoRule    = { 'CUE_A': 'VIS',
                'CUE_C': 'VIS',
                'CUE_B': 'AUD',
                'CUE_D': 'AUD'}

with spa.Network(seed = 100) as model:
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
                                        function=lambda x: x > 0,
                                        label = 'motorClean')

    nengo.Connection(mdRULE.output, pfcRULEmemo.input, transform=5,synapse=0.05)


    with spa.ActionSelection():
        spa.ifmax(0.3*spa.dot(s.NOTHING, pfcCUE),s.NOTHING >> pfcRULEmemo)

        spa.ifmax(spa.dot(s.CUE_A + s.CUE_C, pfcCUE),
                       s.VIS >> pfcRULE)
        spa.ifmax(spa.dot(s.CUE_B + s.CUE_D, pfcCUE),
                       s.AUD >> pfcRULE)
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
