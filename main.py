from convolutionalNetwork import ConvolutionalNetwork
from multilayer_perceptron import MultilayerPerceptron
import numpy as np
from time import time

# Parameters
############################

num_epocs = 25

num_dense = 512

batch_size = 128

load_data = True

save_data = False

path="./models/convolutional_network/best/convolutional_network_512n_30e_f_sigmoid_c_adam.model"
#path='./models/multilayer_perceptron/best/multilayer_perceptron_512n_30e_f_sigmoid_c_adam.model'

save_predictions = True

############################


network = ConvolutionalNetwork('./files', num_epocs, 'ConvolutionalNetwork')
#network = MultilayerPerceptron('./files', num_epocs, 'MultilayerPerceptron')

if load_data: 
    
    network.load_model(num_epocs, num_dense,path=path)
     
else:

    network.layers_construction()
    
    network.model.summary()

    start_time = time()

    network.learn_training(batch_size=batch_size)
    
    elapsed_time = time() - start_time
    
    print("Tiempo empleado: ",elapsed_time)


network.evaluate()

network.model.summary()

if save_predictions:
    
    predictions = network.model.predict(network.data.images_testing)
    
    f = open ('./models/convolutional_network/best/predictions.txt','w')
    #f = open ('./models/multilayer_perceptron/best/predictions.txt','w')
    
    for i in range(0,len(predictions)):
        f.write(repr(np.argmax(predictions[i])))
    
    f.close()  

if save_data:
    network.save_model(num_epocs, num_dense)

