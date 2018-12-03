

import nengo
import nengo_spa as spa
import numpy as np
from robots import CustomRobot
from functools import partial
import actuators
import sensors
import vrep

newCue = True
D = 64
s = spa.sym

# define trial structure =====
trial_duration = 1 #in_sec
cue_length     = 0.2
target_length  = 0.3
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
        self.cues =  ['CUE_A','CUE_B','CUE_C','CUE_D']
        self.targets = ['VIS*RIGHT + AUD*LEFT','VIS*LEFT + AUD*RIGHT']

    def presentCues(self, t):
        if self.trial <= 6:
            # context 1
            cues = ['CUE_A','CUE_B']
        elif 7 <= self.trial <= 15:
            # context 2
            cues = ['CUE_C','CUE_D']
        else:
            # randomized
            cues = ['CUE_A','CUE_B','CUE_C','CUE_D']

        if  (int(t) % self.trial_duration==0) and ( (t-int(t)) == 0):
            self.trial = self.trial + 1
            self.cue_ind = np.random.randint(0,len(cues))

        if  (int(t)%self.trial_duration==0) and (0<= t-int(t) <= self.cue_length):
            return cues[self.cue_ind]
        else:
            return '0'

    def presentTargets(self, t):
        targets = self.targets
        if  (int(t) % self.trial_duration==0) and ( (t-int(t)) == 0):
            self.tar_ind = np.random.randint(0,len(targets))
            print("Trial = {0}, Cue = {1}, Target = {2}".format(self.trial, self.cues[self.cue_ind], self.targets[self.tar_ind]))
        if  (int(t)%self.trial_duration==0) and ( self.tar_ON <= t-int(t) <= (self.tar_ON + self.target_length)):
            return targets[self.tar_ind]
        else:
            return '0'
            
    def simulatetCueTarget(self, t):
        stim = np.array([[-1.5,0.0, 0.2, -1.5,0.0,-0.5, -1.5,0.0,-0.5, -1.5,0.0,-0.5, -1.5,-0.5,-0.5, -1.5,-0.5,-0.5],  # cue A
                         [-1.5,0.0,-0.5, -1.5,0.0, 0.2, -1.5,0.0,-0.5, -1.5,0.0,-0.5, -1.5,-0.5,-0.5, -1.5,-0.5,-0.5],  # cue B
                         [-1.5,0.0,-0.5, -1.5,0.0,-0.5, -1.5,0.0, 0.2, -1.5,0.0,-0.5, -1.5,-0.5,-0.5, -1.5,-0.5,-0.5],  # cue C
                         [-1.5,0.0,-0.5, -1.5,0.0,-0.5, -1.5,0.0,-0.5, -1.5,0.0, 0.2, -1.5,-0.5,-0.5, -1.5,-0.5,-0.5],  # cue D
                         [-1.5,0.5,-0.5, -1.5,0.5,-0.5, -1.5,0.5,-0.5, -1.5,0.5,-0.5, -1.5,-1.0, 0.2, -1.5, 1.0, 0.2],  # Target 1
                         [-1.5,0.5,-0.5, -1.5,0.5,-0.5, -1.5,0.5,-0.5, -1.5,0.5,-0.5, -1.5, 1.0, 0.2, -1.5,-1.0, 0.2],  # Target 2
                         [-1.5,0.5,-0.5, -1.5,0.5,-0.5, -1.5,0.5,-0.5, -1.5,0.5,-0.5, -1.5,-0.5,-0.5, -1.5,-0.5,-0.5]]) # all hidden
                
        if  0.001< t-int(t) <= self.cue_length:
            return stim[self.cue_ind,:]
            
        elif self.tar_ON+0.01 < t-int(t) <= (self.tar_ON + self.target_length):
            return stim[self.tar_ind+4,:]
        else:
            return stim[-1,:]    


# Dimensions of the SPA vocabulary
#DIM = 16 #32
# vocabulary
vocab = spa.Vocabulary(D)
vocab.populate('CUE_A; CUE_B; CUE_C; CUE_D; RIGHT; LEFT; VIS; AUD; NOTHING; HOME')
assoRule    = { 'CUE_A': 'VIS',
                'CUE_C': 'VIS',
                'CUE_B': 'AUD',
                'CUE_D': 'AUD'}
                
#vocab_moves = spa.Vocabulary(1)
#vocab_moves.add('HOME',(0))
#vocab_moves.add('RIGHT',(-1,))
#vocab_moves.add('LEFT',(1,))
moves       = ['HOME','RIGHT','LEFT']
movesMap    = {k : k for k in moves}
#movesMap    = dict(zip(moves, coordinates))

# Robot
vrepc = CustomRobot(sim_dt=0.01, nengo_dt=0.001, sync=True)
# add boxes
vrepc.add_actuator("boxCueA", actuators.position, dim=3)
vrepc.add_actuator("boxCueB", actuators.position, dim=3)
vrepc.add_actuator("boxCueC", actuators.position, dim=3)
vrepc.add_actuator("boxCueD", actuators.position, dim=3)
vrepc.add_actuator("boxTarget_1", actuators.position, dim=3)
vrepc.add_actuator("boxTarget_2", actuators.position, dim=3)
# robot Left and right wheels
vrepc.add_actuator("MTB_Robot", actuators.orientation, dim=3)
#vrepc.add_actuator("Pioneer_p3dx_leftMotor", actuators.joint_velocity)
#vrepc.add_actuator("Pioneer_p3dx_rightMotor", actuators.joint_velocity)

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
                                        function=lambda x: 1 if x > 0.1 else 0,
                                        label='pfcRULEasso')
    stim_cue >> pfcRULEasso
    pfcRULEasso >> pfcRULEmemo

    # connections
    stim_cue >> ppc
    stim_cue >> pfcCUE

    motorClean  = spa.WTAAssocMem(threshold=0.1, input_vocab=vocab,
                                        output_vocab=vocab,
                                        mapping = movesMap,
                                        function=lambda x: 1 if x > 0.1 else 0,
                                        label = 'motorClean')
    
    nengo.Connection(mdRULE.output, pfcRULEmemo.input, transform=5,synapse=0.05)

    with spa.ActionSelection() as act_sel:
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
    
    # Confidence of the cotex
    cueVectors = vocab.parse('CUE_A+CUE_B+CUE_C+CUE_D').v
    contexConfidence = nengo.Ensemble(100,1,label='contexConfidence')
    nengo.Connection(mdCUE.output, contexConfidence, transform=[cueVectors])
     
    with spa.Network(label='Pioneer p3dx', seed=13) as modelR:

        #vrepN    = nengo.Node(vrepc, size_in=20, size_out=0)
        nodeVrep = vrepc.build_node()
        
        # Control boxes
        posBoxes = nengo.Node(exp.simulatetCueTarget)  
        nengo.Connection(posBoxes, nodeVrep[0:18])
        #posBoxes = nengo.Node([0.5,0.5,-0.5, 0.5,0.5,-0.5, 0.5,0.5,-0.5, 
        #                       0.5,0.5,-0.5, -0.5,-0.5,-0.5, -0.5,-0.5,-0.5])  
        
        # Control Robots
        def get_direction(x):
            gain = 1
            if spa.dot(vocab.parse('LEFT').v, x) > 0.35:
                print('left')
                return spa.dot(vocab.parse('LEFT').v, x) * gain 
            elif spa.dot(vocab.parse('RIGHT').v, x) > 0.35:
                print('right')
                return spa.dot(vocab.parse('RIGHT').v, x) * -gain
            else:
                return 0.0
                
        outputMotor = nengo.Ensemble(1,D, neuron_type=nengo.Direct())
        #passDirection = nengo.Ensemble(n_neurons=500,dimensions=1, radius=20)
    
        # Connections to robots
        nengo.Connection(motorClean.output, outputMotor,synapse=None)
        #nengo.Connection(outputMotor, passDirection, function=get_direction)
        #nengo.Connection(passDirection, nodeVrep[-1], synapse=None)
        nengo.Connection(outputMotor, nodeVrep[-1], function=get_direction)#, synapse=None)
        #nengo.Node([])
        
        #visNode = nengo.Ensemble(n_neurons=1, dimensions=1, neuron_type=nengo.Direct())
        #nengo.Connection(outputMotor, visNode, function=get_direction, synapse=None)

    