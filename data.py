from mnist import MNIST
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


"""
 Clase que representa al conjunto de datos
"""

class Dataset:
    
    def __init__(self, data_source):
        
        self.load_data(data_source)
        
        self.array_convert()
        
        self.images_array_reshape()
        
        self.array_normalize(1)
    
    
    def load_data(self,data_source):
        self.mndata = MNIST(data_source)
        
        # Training data
        self.images_training = self.mndata.load_training()[0]
        self.labels_training = self.mndata.load_training()[1]
        
        # Testing data
        self.images_testing = self.mndata.load_testing()[0]
        self.labels_testing = self.mndata.load_testing()[1]
    
    def array_convert(self):
        self.images_training = np.asarray(self.images_training)
        self.labels_training = np.asarray(self.labels_training)
        
        self.images_testing = np.asarray(self.images_testing)
        self.labels_testing = np.asarray(self.labels_testing)
        
    def images_array_reshape(self):
        self.images_training = self.images_training.reshape(self.images_training.shape[0], 28,28)
        self.images_testing = self.images_testing.reshape(self.images_testing.shape[0], 28,28)
    
    
    def array_normalize(self,_axis):
        self.images_training = tf.keras.utils.normalize(self.images_training, axis = _axis)  
        self.images_testing = tf.keras.utils.normalize(self.images_testing, axis = _axis)
        
    def print_numbber(self, number):
        plt.imshow(number, cmap = 'gray')
        plt.show()
    