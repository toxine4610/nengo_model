
import numpy as np

time = np.array([0,1,2,4,5,6,7,8,9,10,11,12])

trial_counter = 0
# present cues every 2s for 200ms

def presentCues(t):
    t_curr = t*1000
    if t_curr % 2000 == 0:
        trial_counter = trial_counter+1
    return trial_counter
    # print(trial_counter)
    # def f(t):
    #     if 0+trial_counter < t < 0.2+trial_counter:
    #         return 'CUE_A'
    #     else:
    #         return '0'
    # return f

for i in time:
    f = presentCues(i)
    print('t' + str(i) )
