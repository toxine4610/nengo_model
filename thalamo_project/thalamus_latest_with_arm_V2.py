
'''
  thalamus latest with arm.
  version 2
'''

import nengo
import nengo_spa as spa
import numpy as np
from osim.env import OsimEnv
import scipy.optimize
import time


newCue = True
D = 64
s = spa.sym

# define trial structure =====
trial_duration = 1 #in_sec
cue_length     = 0.2
target_length  = 0.2
tar_ON         = 0.6 # delay period is 600 ms


class MuscleEnv(OsimEnv):
   # we will use the models from the osim_rl library as they have already the
   # Geometry directlry with nice 3D bones and will automatically be shown in
   # the simulator you don't need to copy any other geometry files in the
   # working directory
   model_path = 'MoBL_ARMS_J_Simple_032118.osim'

   def reward(self):
       return 0

# NOTE: should the step size be smaller?
timestep = 0.01

env = MuscleEnv()#integrator_accuracy=1e-2)
env.osim_model.stepsize = timestep
env.osim_model.model.getVisualizer().getSimbodyVisualizer().setGroundHeight(-1.0)
env.reset()
visualizer = env.osim_model.model.updVisualizer()

env2 = MuscleEnv(visualize=False)#, integrator_accuracy=1e-2)
env2.osim_model.stepsize = timestep
env2.reset()


env.step([0.05]*50)
time.sleep(0.01)
env2.step([0.05]*50)

def compare_angles(angles, des_xyz):
    # set the state of env2 to the state of env1
    env2.osim_model.model.setStateVariableValues(
        env2.osim_model.state,
        env.osim_model.model.getStateVariableValues(
            env.osim_model.state))

    # names = env.osim_model.model.getStateVariableNames()
    # for ii in range(names.getSize()):
    #     print(names.get(ii))

    # set the joint angles to the angles specified by the optimizer
    for ii, name in enumerate([
            'shoulder0/elv_angle/value',
            'shoulder1/shoulder_elv/value',
            'shoulder1/shoulder1_r2/value',
            'shoulder2/shoulder_rot/value',
            'elbow/elbow_flexion/value']):
        env2.osim_model.model.setStateVariableValue(
            state=env2.osim_model.state,
            name=name,
            value=angles[ii])

    # get the (x,y,z) of the hand with specified angles and calculate
    # distance to desired (x,y,z)
    env2.osim_model.model.realizeAcceleration(env2.osim_model.state)
    marker_set = env2.osim_model.model.getMarkerSet()
    hand_xyz = marker_set.get(7).getLocationInGround(env2.osim_model.state)
    # hand_xyz = env2.osim_model.model.getStateVariableValue(
    #     env2.osim_model.state,
    #     'markers/Handle/pos') # TODO: confirm this is the actual hand position
    # for ii in range(marker_set.getSize()):
    #     marker = marker_set.get(ii)
    #     print('marker name: ', marker.getName())
    #     # hand_xyz = [marker.getLocationInGround(env2.osim_model.state)[ii]
    #     #             for ii in range(3)]
    #     pos = marker.getLocationInGround(env2.osim_model.state)
    #     print('marker pos: ', pos)
    cost = 0.0
    for ii in range(len(des_xyz)):
        cost += (hand_xyz[ii] - des_xyz[ii])**2

    return cost

def calculate_angles(des_xyz):

    return scipy.optimize.minimize(
        fun=compare_angles, args=(des_xyz,),
        x0=np.zeros(5),
        bounds=[[0, 1]] * 5,
        method='SLSQP',
        # options={
        #     #'disp':True}
        #     # 'eps':1e-2,
        #     }
        )['x']

def mapdirection_to_scalar(x):
    # returns values in degrees.
    if np.dot( vocab.parse('LEFT').v, x ) >= 0.3:
        return [2,1.5,1.5]
    elif np.dot(vocab.parse('RIGHT').v, x) >= 0.3:
        return [1,1.5,1.5]
    else:
        return [1.5,1.5,1.5]

def output_fcn(t, x):
    if int(t*1000)%1 == 0:
        time.sleep(.01)
        foo = mapdirection_to_scalar(x)
        angles = calculate_angles(foo)
        print([float('%.3f' % a) for a in angles])

        # set the joint angles to the angles specified by the optimizer
        for ii, name in enumerate([
                'shoulder0/elv_angle/value',
                'shoulder1/shoulder_elv/value',
                'shoulder1/shoulder1_r2/value',
                'shoulder2/shoulder_rot/value',
                'elbow/elbow_flexion/value']):
            env.osim_model.model.setStateVariableValue(
                state=env.osim_model.state,
                name=name,
                value=angles[ii])
        visualizer.updSimbodyVisualizer().drawFrameNow(env.osim_model.state)

    return angles

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

    # map motorClean output to a scalar.........................................
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
    # nengo.Connection( turn_direction, env[65], synapse  = None)

    # feed input to arm..
    # connect turn_direction  -->> output
    des_xyz = nengo.Node([1.5, 1.5])

    output = nengo.Node(output_fcn, size_in=3, size_out=5)
    nengo.Connection(des_xyz, output)
