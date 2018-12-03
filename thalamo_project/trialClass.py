

class Experiment (object):
    """docstring for ."""
    def __init__(self, trial_duration, cue_length, target_length, tar_ON):
        self.trial_counter = 0
        self.trial_duration = trial_duration
        self.cue_length  = cue_length
        self.target_length = target_length
        self.tar_ON   = tar_ON

    def presentCues(self, t):
        if  (int(t)%self.trial_duration==0)  and (0<= t-int(t) <= self.cue_length):
            self.trial_counter = self.trial_counter + 1
            return 100
        else:
            return 0

    def presentTargets(self, t):
        if  (int(t)%self.trial_duration==0)  and ( self.tar_ON <= t-int(t) <= (self.tar_ON + self.target_length)):
            self.trial_counter = self.trial_counter + 1
            return 100
        else:
            return 0




import numpy as np
exp = Experiment()
time = np.array([0,1,2,4,5,6,7,8,9,10,11,12])
for i in time:
    print( exp.presentCues(i))
