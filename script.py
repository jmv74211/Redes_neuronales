from convolutionalNetwork import ConvolutionalNetwork
from multilayer_perceptron import MultilayerPerceptron
from time import time

max_num_epocs = 30
num_dense = 512
f_activation='sigmoid'

# Generamos el archivo con los resultados obtenidos

f = open ('./results/multilayer_perceptron/' + f_activation + '/multilayer_perceptron_'  + repr(num_dense) + 'n_' + repr(max_num_epocs) 
          + 'e_f_' + f_activation + '.txt','w')

for i in range(1,max_num_epocs+1):
    
    print("-"*50)
    print("Construye red neuronal con ", i, "épocas de entrenamiento")
    print("-"*50)
    
    #network = ConvolutionalNetwork('./files', i, 'ConvolutionalNetwork')
    
    network = MultilayerPerceptron('./files', i, 'MultilayerPerceptron')
    
    network.layers_construction(training_sample = True)
    
    network.model.summary()
    
    start_time = time()
    
    network.learn_training_sample()
    
    elapsed_time = time() - start_time

    loss,acc = network.evaluate_training_sample()
    
    f.write(repr(i) + '\t' + repr(loss) + '\t' + repr(acc) + '\t' + repr(elapsed_time) + '\n')
    
f.close()

network.save_model(num_epocs=max_num_epocs,num_dense=num_dense,path='./models/multilayer_perceptron/' + f_activation + '/multilayer_perceptron_' + repr(num_dense) + 'n_' + repr(max_num_epocs) 
        + 'e_f' + f_activation + '.model')

network.model.summary()

#save_fig(max_num_epocs)




