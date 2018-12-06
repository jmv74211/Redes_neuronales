import numpy as np
from matplotlib import pyplot as plt

def save_fig(num_epocs):

#num_epocs = 6

    file='./results/multilayer_perceptron/multilayer_perceptron_' + repr(num_epocs) + 'e.txt'
    
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
    
    plt.savefig('./results/multilayer_perceptron/multilayer_perceptron_' + repr(num_epocs) + 'e_acc.png')
    
    plt.figure() 
    plt.plot(epocas,loss)
    plt.title("Variación de loss respecto a épocas")
    plt.xlabel("Loss")
    plt.ylabel("Número de épocas")
    
    
    manager = plt.get_current_fig_manager() 
    manager.resize(*manager.window.maxsize()) 
    
    plt.savefig('./results/multilayer_perceptron/multilayer_perceptron_' + repr(num_epocs) + 'e_loss.png')






