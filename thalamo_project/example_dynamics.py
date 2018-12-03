
import numpy as np
import nengolib
import nengo
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from nengolib import Network
from nengolib.signal import Hankel, s
from nengo.processes import WhiteSignal, PresentInput

processes = nengo.processes.WhiteSignal(10, high=10, y0 = 0)
neuron_type = nengo.LIF()

def function(w):
    return w.sum()
    
def inputfunction(t):
    return np.cos(8*t)

pulse_s = 0
pulse_w = 1.5
pulse_h = 1.0
    
T = 6.0
dt = 0.001
pulse = np.zeros(int(T/dt))
pulse[int(pulse_s/dt):int((pulse_s + pulse_w)/dt)] = pulse_h


with nengolib.Network() as model:
    rw = nengolib.networks.RollingWindow(
        theta = 2, n_neurons = 600, process = processes,
        neuron_type=neuron_type)
    stim = nengo.Node(PresentInput(pulse, dt))
  
    nengo.Connection(stim, rw.input, synapse = None)
    output = rw.add_output(function=function)
