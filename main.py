from neuronalNetwork import NeuronalNetwork
from convolutionalNetwork import ConvolutionalNetwork
from multilayer_perceptron import MultilayerPerceptron

     
# Parameters
############################

num_epocs = 30

num_dense = 512

load_data = True

save_data = False

############################


network = ConvolutionalNetwork('./files', num_epocs, 'ConvolutionalNetwork')
#network = MultilayerPerceptron('./files', num_epocs, 'ConvolutionalNetwork')

if load_data: 
    
    network.load_model(num_epocs, num_dense)
     
else:

    network.layers_construction()

    network.learn()


network.evaluate()

network.model.summary()

if save_data:
    network.save_model(num_epocs, num_dense)

