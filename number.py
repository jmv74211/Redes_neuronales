import matplotlib.pyplot as plt
import numpy as np


"""
 Clase que representa a un número
"""

class number:
    def __init__(self, dataNumber):
        self.data = self.array_convert(dataNumber)
        
     
    def array_convert(self,dataNumber):
        
        # Convierto la lista a vector
        number_array= np.asarray(number)
    
        # Convierto el vector 784 en matriz 28x28
        number_array=number_array.reshape(28,28)
        
        return number_array
        
    def print(self):
        plt.imshow(self.dataNumber, cmap = 'gray')
        plt.show()    
  
    