import numpy as np
from matplotlib import pyplot as plt




#  GRÁFICO DE BARRAS 
num_epocs = 30

file='./results/multilayer_perceptron/data.txt'

acc= np.loadtxt(file, delimiter='\t', skiprows=0,usecols=[0])
acc2= np.loadtxt(file, delimiter='\t', skiprows=0,usecols=[1])
acc3= np.loadtxt(file, delimiter='\t', skiprows=0,usecols=[2])
acc4= np.loadtxt(file, delimiter='\t', skiprows=0,usecols=[3])
acc5= np.loadtxt(file, delimiter='\t', skiprows=0,usecols=[4])
acc6= np.loadtxt(file, delimiter='\t', skiprows=0,usecols=[5])


lista_datos = [acc,acc2,acc3,acc4,acc5,acc6]
lista_labels = ["RA", "SA", "RR", "SR", "RS", "SS"]
xx = range(1,len(lista_datos)+1)

fig = plt.figure() 
ax = fig.add_subplot(111)
ax.bar(xx,lista_datos,width=0.5,align="center", color=(1,0,0))
ax.set_xticks(xx)
ax.set_xticklabels(lista_labels)
ax.set_ylabel("Acc")
ax.set_xlabel("FunciónActivación-Optimizador")
ax.set_ylim(0.9,0.99)

manager = plt.get_current_fig_manager() 
manager.resize(*manager.window.maxsize()) 

plt.savefig('./comparativa.png')



