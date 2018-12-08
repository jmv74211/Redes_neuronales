from neuronalNetwork import NeuronalNetwork
from convolutionNetwork import ConvolutionNetwork
from multilayer_perceptron import MultilayerNetwork


num_epocs = 25

num_dense = 256

network = ConvolutionNetwork('./files',num_epocs)

#network.load_model('./models/convolutional_network/convolutional_network_' + repr(num_dense) + 'n_' + repr(num_epocs) +'e.model')

network.layers_construction()

network.learn()

network.evaluate()

network.model.summary()

network.save_model('./models/convolutional_network/convolutional_network_' + repr(num_dense) + 'n_' + repr(num_epocs) +'e.model')

