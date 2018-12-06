from neuronalNetwork import NeuronalNetwork


max_num_epocs = 6

# Generamos el archivo con los resultados obtenidos

f = open ('./results/perceptron_multilayer_' + repr(max_num_epocs) + 'e.txt','w')

for i in range(1,max_num_epocs+1):
    print("Época número ", i)
    
    network = network = NeuronalNetwork('./files',i)
    
    network.layers_construction()

    network.learn()

    loss,acc = network.evaluate()
    
    f.write(repr(i) + '\t' + repr(loss) + '\t' + repr(acc) + '\n')

f.close()



