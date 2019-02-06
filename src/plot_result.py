import numpy as np
from matplotlib import pyplot as plt

num_epocs = 30

#file='./results/multilayer_perceptron/multilayer_perceptron_' + repr(num_epocs) + 'e.txt'
file='./results/multilayer_perceptron/training/tanh/multilayer_perceptron_128n_30e_f_tanh.txt'


epocas = np.loadtxt(file, delimiter='\t', skiprows=0,usecols=[0])
loss= np.loadtxt(file, delimiter='\t', skiprows=0,usecols=[1])
acc= np.loadtxt(file, delimiter='\t', skiprows=0,usecols=[2])

plt.figure() 
plt.plot(epocas,acc)
plt.title("Variación de acc respecto a épocas")
plt.xlabel("Acc")
plt.ylabel("Número de épocas")


manager = plt.get_current_fig_manager() 
manager.resize(*manager.window.maxsize()) 

plt.savefig('./results/multilayer_perceptron/training/multilayer_perceptron_' + repr(num_epocs) + 'e_acc.png')

plt.figure() 
plt.plot(epocas,loss)
plt.title("Variación de loss respecto a épocas")
plt.xlabel("Loss")
plt.ylabel("Número de épocas")


manager = plt.get_current_fig_manager() 
manager.resize(*manager.window.maxsize()) 

plt.savefig('./results/multilayer_perceptron/training/multilayer_perceptron_' + repr(num_epocs) + 'e_loss.png')






