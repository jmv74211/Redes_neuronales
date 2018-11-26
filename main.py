from data import Dataset
from neuronalNetwork import NeuronalNetwork

network = NeuronalNetwork('./archivos',5)

network.layers_construction()
network.learn()

network.data.print_numbber(network.data.images_testing[63])