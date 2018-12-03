

import nengo
import nengo_spa as spa
import numpy as np

model = spa.Network()
dims  = 32
# n_Trials = 100

# stimsC1 = ['HP','LP']
# stimsC1 = stimsC1*n_Trials

# stimsC2 = ['UV','GR']
# stimsC2 = stimsC2*n_Trials

with model:

    sensory = spa.State(dims, label = 'sensory')
    motor = spa.State(dims, label = 'motor')

    def cueFN(t):
        if 0.1<t<0.2:
            return 'CUE_A'
        elif 0.8<t<0.9:
            return 'AUD*LEFT + VIS*RIGHT'
        else:
            return '0'

    stim_type = spa.Transcode( cueFN, output_vocab = sensory.vocab)
    stim_type >> sensory
