import nengo
import nengo_spa as spa
import numpy as np
s = spa.sym

D = 32
dims = D
pfcFB = 0.1

def target_input(t):
    targets     = ['SOUND*RIGHT + LIGHT*LEFT','SOUND*LEFT + LIGHT*RIGHT']
    t_onTrial   = t - int(t)
    if  0.6 < t_onTrial < 0.8:
        return targets[np.random.randint(0,2)]
    else:
        return '0'

def cue_input(t):
    cues        = ['CUE_A', 'CUE_B', 'CUE_C', 'CUE_D']
    t_onTrial   = t - int(t)
    if 0 < t_onTrial < 0.2:
        return cues[np.random.randint(0,4)]
    else:
        return '0'

map = {
    'CUE_A': 'CONTEXT1',
    'CUE_B': 'CONTEXT1',
    'CUE_C': 'CONTEXT2',
    'CUE_D': 'CONTEXT2',
    }


model = spa.Network()
with model:
    # input
    target_in = spa.Transcode(target_input, output_vocab = D, label='target_input')
    cue_in    = spa.Transcode(cue_input, output_vocab  = D, label='cue_input')

    # state representation of input
    target  = spa.State(D, label='Target')
    cue     = spa.State(D, label='Cue')
    motor = spa.State(dims, feedback = 1, label = 'motor')


    # Connect to repreesentation of inputs
    target_in   >> target
    cue_in      >> cue

    # pfc
    pfc  = spa.State(dims, feedback = pfcFB,  label = 'pfc')
    vocab = pfc.vocab
    vocab.add('RULE1', vocab.parse('CUE_A+CUE_C'))
    vocab.add('RULE2', vocab.parse('CUE_B+CUE_D'))
    vocab.add('CONTEXT1', vocab.parse('CUE_A+CUE_B'))
    vocab.add('CONTEXT2', vocab.parse('CUE_C+CUE_D'))

    cue >> pfc

    with spa.ActionSelection():
        spa.ifmax( 0.5, s.X*0 >> pfc)
        spa.ifmax( 0.8*spa.dot(pfc, s.RULE1) +  0.8*spa.dot(target, s.LIGHT*(s.RIGHT+s.LEFT)),
                           target*~s.LIGHT >> motor )
        spa.ifmax( 0.8*spa.dot(pfc, s.RULE2) +  0.8*spa.dot(target, s.SOUND*(s.RIGHT+s.LEFT)),
                           target*~s.SOUND >> motor )

    md = spa.WTAAssocMem(threshold = 0.5, input_vocab = pfc.vocab, mapping = map, label = 'MD')
    pfc >> md
    md*0.8 >> pfc
