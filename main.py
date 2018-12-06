from neuronalNetwork import NeuronalNetwork

num_epocs = 6

network = NeuronalNetwork('./files',num_epocs)

#network.load_model('./models/multilayer_perceptron_' + repr(num_epocs) +'e.model')

network.layers_construction()

network.learn()

network.evaluate()

#network.model.summary()

#network.save_model('./models/multilayer_perceptron_' + repr(num_epocs) +'e.model')