## using spa vocab, in html....
import nengo
import nengo_spa as spa
import numpy as np


# Dimensions of the SPA vocabulary
DIM = 16 #32

# Define the vocabulary of semantic pointers that will be used
location_vocab   = spa.Vocabulary(dimensions=DIM)
coordinate_vocab = spa.Vocabulary(dimensions=1)

# Locations that the quadcopter knows about
locations = ['LEFT', 'RIGHT']
coordinates = [(1),(-1)]
coordinate_names = ['C'+l for l in locations]

# Generate a semantic pointer for each of these locations in the vocabulary
for location in locations:
    location_vocab.parse(location)

# Generate a vector for each coordinate
for i, coordinate in enumerate(coordinates):
    coordinate_vocab.add(coordinate_names[i], coordinate)


model = spa.Network(label='Navigation Quadcopter', seed=13)
with model:
    model.assoc_mem = spa.AssociativeMemory(input_vocab=location_vocab,
                                            output_vocab=coordinate_vocab,
                                            input_keys=locations,
                                            output_keys=coordinate_names,
                                            wta_output=True,
