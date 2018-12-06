from neuronalNetwork import NeuronalNetwork
from plot_result import save_fig


max_num_epocs = 30

# Generamos el archivo con los resultados obtenidos

f = open ('./results/multilayer_perceptron/multilayer_perceptron_' + repr(max_num_epocs) + 'e.txt','w')

for i in range(1,max_num_epocs+1):
    
    print("-"*50)
    print("Construye red neuronal con ", i, "épocas de entrenamiento")
    print("-"*50)
    
    network = NeuronalNetwork('./files',i)
    
    network.layers_construction()

    network.learn()

    loss,acc = network.evaluate()
    
    f.write(repr(i) + '\t' + repr(loss) + '\t' + repr(acc) + '\n')
    
f.close()

network.save_model('./models/multilayer_perceptron/multilayer_perceptron_' + repr(max_num_epocs) +'e.model')

save_fig(max_num_epocs)




