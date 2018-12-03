

import nengo
import numpy as np
from nengo.processes import Piecewise


model = nengo.Network()

with model:

    stimAinput = Piecewise({0: 0, 1 : 1, 2: 0, 3: 1, 4: 0, 5: 1, 6:0, 7:0, 8:0,9:0,10:0})
    stimBinput = Piecewise({0: 1, 1 : 0, 2: 1, 3: 0, 4: 1, 5: 0, 6:0, 7:0, 8:0,9:0,10:0})

    stimCinput = Piecewise({0: 0, 1 : 0, 2: 0, 3: 0, 4: 0, 5: 0, 6:1, 7:0, 8:1, 9:0, 10:1})
    stimDinput = Piecewise({0: 0, 1 : 0, 2: 0, 3: 0, 4: 0, 5: 0, 6:0, 7:1, 8:0, 9:1, 10:0})

    stimA = nengo.Node( stimAinput )
    stimB = nengo.Node( stimBinput )
    stimC = nengo.Node( stimCinput )
    stimD = nengo.Node( stimDinput )

    ensPreA  = nengo.Ensemble(100, dimensions = 1)
    ensPreB =  nengo.Ensemble(100, dimensions = 1)
    ensPreC  = nengo.Ensemble(100, dimensions = 1)
    ensPreD =  nengo.Ensemble(100, dimensions = 1)

    ensPost1 = nengo.Ensemble(100, dimensions = 1, radius = 1.4)
    ensPost2 = nengo.Ensemble(100, dimensions = 1, radius = 1.4)

    nengo.Connection(stimA, ensPreA)
    nengo.Connection(stimB, ensPreB)
    nengo.Connection(stimC, ensPreC)
    nengo.Connection(stimD, ensPreD)


    inhibit = nengo.Node(stimCinput)
    nengo.Connection(inhibit, ensPost1.neurons, transform=np.ones((ensPost1.n_neurons, 1)) * 10, synapse=None)

    inhibit2 = nengo.Node(stimAinput)
    nengo.Connection(inhibit2, ensPost2.neurons, transform=np.ones((ensPost2.n_neurons, 1)) * 10, synapse=None)

    # how do I compute a similarity trace????
    # stop learning once the stimularity crosses some threshold value

    connA = nengo.Connection(ensPreA, ensPost1,
    solver=nengo.solvers.LstsqL2(weights=True))

    connA.learning_rule_type = nengo.BCM(learning_rate=5e-10)

    connB = nengo.Connection(ensPreB, ensPost1,
    solver=nengo.solvers.LstsqL2(weights=True))

    connB.learning_rule_type = nengo.BCM(learning_rate=5e-10)

    connC = nengo.Connection(ensPreC, ensPost2,
    solver=nengo.solvers.LstsqL2(weights=True))

    connC.learning_rule_type = nengo.BCM(learning_rate=5e-10)

    connD = nengo.Connection(ensPreD, ensPost2,
    solver=nengo.solvers.LstsqL2(weights=True))

    connD.learning_rule_type = nengo.BCM(learning_rate=5e-10)
