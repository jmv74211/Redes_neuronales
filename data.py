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
  
        # Training sample
        tam_sample = int( (self.mndata.load_training()[0].__len__()/4)*3)
        
        self.images_training_sample = self.images_training[0:tam_sample]
        self.labels_training_sample = self.labels_training[0:tam_sample]
    
        # Testing data
        self.images_testing = self.mndata.load_testing()[0]
        self.labels_testing = self.mndata.load_testing()[1]
        
        
        # Testing training sample
        self.images_testing_sample = self.images_training[tam_sample:self.images_training.__len__()]
        self.labels_testing_sample = self.labels_training[tam_sample:self.images_training.__len__()]
        
    def array_convert(self):
        
        # Training data
        self.images_training = np.asarray(self.images_training)
        self.labels_training = np.asarray(self.labels_training)
        
        # Testing data
        self.images_testing = np.asarray(self.images_testing)
        self.labels_testing = np.asarray(self.labels_testing)
        
        # Training sample
        self.images_training_sample = np.asarray(self.images_training_sample)
        self.labels_training_sample = np.asarray(self.labels_training_sample)
        
        # Testing training sample
        self.images_testing_sample = np.asarray(self.images_testing_sample)
        self.labels_testing_sample = np.asarray(self.labels_testing_sample)
         
    def images_array_reshape(self):
        
        # Training, testing data
        self.images_training = self.images_training.reshape(self.images_training.shape[0], 28,28,1)
        self.images_testing = self.images_testing.reshape(self.images_testing.shape[0], 28,28,1)
        
        # Training, testing sample data
        self.images_training_sample = self.images_training_sample.reshape(self.images_training_sample.shape[0], 28,28,1)
        self.images_testing_sample = self.images_testing_sample.reshape(self.images_testing_sample.shape[0], 28,28,1)
        
    def array_normalize(self,_axis):
        
        # Training, testing data
        self.images_training = tf.keras.utils.normalize(self.images_training, axis = _axis)  
        self.images_testing = tf.keras.utils.normalize(self.images_testing, axis = _axis)
        
        # Training, testing sample data
        self.images_training_sample = tf.keras.utils.normalize(self.images_training_sample, axis = _axis)  
        self.images_testing_sample = tf.keras.utils.normalize(self.images_testing_sample, axis = _axis)
        
    def print_numbber(self, number):
        plt.imshow(number, cmap = 'gray')
        plt.show()
    