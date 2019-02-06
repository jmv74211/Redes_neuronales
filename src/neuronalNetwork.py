from data import Dataset
from tensorflow.python.keras.models import load_model

from keras.models import Sequential
from abc import ABC, abstractmethod


class NeuronalNetwork(ABC):
    
    def __init__(self, data_source, epocs, neuronal_network_type, num_clases = 10):
        
        # Número de épocas
        self.epocs = epocs
        
        # Número de neuronas de salida
        self.num_classes = num_clases
        
        # Cargo los datos
        self.data = Dataset(data_source,neuronal_network_type)
        
        # Creo el modelo
        self.model = Sequential()
        
        #Clase abstracta no instanciable
        super().__init__()
        
           
    @abstractmethod
    def layers_construction(self, training_sample = False): pass
        
    @abstractmethod 
    def save_model(self, num_epocs, num_dense, path=None): pass
    
    @abstractmethod   
    def load_model(self, num_epocs, num_dense, path=None): pass    
                  
    def learn_training_sample(self, batch_size=None):
        
        self.model.fit(self.data.images_training_sample, self.data.labels_training_sample, epochs=self.epocs, batch_size=batch_size)
    
    def learn(self, batch_size=None):
        
        self.model.fit(self.data.images_training, self.data.labels_training, epochs=self.epocs, batch_size=batch_size)
    
    def evaluate_training_sample(self):
        
        loss, acc = self.model.evaluate(self.data.images_testing_sample,self.data.labels_testing_sample)
        print("loss = " , loss , " || acc = ", acc)
        
        return (loss,acc)
    
    def evaluate(self):
        
        loss, acc = self.model.evaluate(self.data.images_testing,self.data.labels_testing)
        print("loss = " , loss , " || acc = ", acc)
        
        return (loss,acc)
        
    def model_summary(self):
        self.model.summary()
        
        