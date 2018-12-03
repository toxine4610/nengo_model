
import nengo
import nengo_spa as spa
import numpy as np
from nengo_extras.learning_rules import AML

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show

model = spa.Network()
dims  = 32

s = spa.sym

pfcFB = 0.1

###############################################################################
# cheating approach....
map = {
    'CUE_A': 'CONTEXT1',
    'CUE_B': 'CONTEXT1',
    'CUE_C': 'CONTEXT2',
    'CUE_D': 'CONTEXT2',
    }
###############################################################################

with model:

    # define populations...
    cues    = spa.State(dims, label = 'cues')
    targets = spa.State(dims, label =  'targets')
    motor   = spa.State(dims, feedback = 1, label = 'motor')
    pfc     = spa.State(dims, feedback = pfcFB,  label = 'pfc')
    contextEncoder = spa.State(dims, feedback = 1, label = 'contextEncoder')

    def cuesFN(t):
        if 0.1<t<0.2:
            return 'CUE_B'
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
    vocab.add('RULE1', vocab.parse('CUE_A+CUE_C'))  # go to VIS
    vocab.add('RULE2', vocab.parse('CUE_B+CUE_D'))  # go to SOUND
    # vocab.add('CONTEXT1', vocab.parse('CUE_A+CUE_B'))
    # vocab.add('CONTEXT2', vocab.parse('CUE_C+CUE_D'))

    cues >> pfc
    pfc >> contextEncoder

    with spa.ActionSelection():
        spa.ifmax( 0.5, s.X*0 >> pfc)
        spa.ifmax( 0.8*spa.dot(pfc, s.RULE1) +  0.8*spa.dot(targets, s.VIS*(s.RIGHT+s.LEFT)),
                       targets*~s.VIS >> motor )
        spa.ifmax( 0.8*spa.dot(pfc, s.RULE2) +  0.8*spa.dot(targets, s.AUD*(s.RIGHT+s.LEFT)),
                       targets*~s.AUD >> motor )



    # md = spa.WTAAssocMem(threshold = 0.5, input_vocab = pfc.vocab, mapping = map, label = 'MD')
    # pfc >> md
    # md*0.8 >> pfc

    #pre = nengo.Ensembles(

    #md = nengo.Node(size_in = dims + 2)

    #c = nengo.Connection( pfc.output, md, learning_rule_type = AML(dims), function = lambda x: np.zeros(dims) )
