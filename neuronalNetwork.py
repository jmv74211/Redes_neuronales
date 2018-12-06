from data import Dataset
import tensorflow as tf
from tensorflow.python.keras.models import load_model

class NeuronalNetwork:
    
    def __init__(self, data_source,_epocs):
        
        # Número de épocas
        self.epocs = _epocs
        
        # Cargo los datos
        self.data = Dataset(data_source)
        
        # Creo el modelo
        self.model = tf.keras.models.Sequential()
           
    
    def layers_construction(self, training_sample = False):
        
        # Aplana la matriz de datos (números)
        if not training_sample:
            self.model.add(tf.keras.layers.Flatten(input_shape=self.data.images_training[0].shape))
        else:
            self.model.add(tf.keras.layers.Flatten(input_shape=self.data.images_training_sample[0].shape))
        
        # 128 neuronas, función activación
        self.model.add(tf.keras.layers.Dense(128,tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128,tf.nn.relu))
        
        # Capa de salida
        self.model.add(tf.keras.layers.Dense(10,tf.nn.softmax))
        
        self.model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
                       
    def learn_training_sample(self):
        
        self.model.fit(self.data.images_training_sample,self.data.labels_training_sample, epochs=self.epocs)
    
    def learn(self):
        
        self.model.fit(self.data.images_training,self.data.labels_training, epochs=self.epocs)
    
    def evaluate_training_sample(self):
        
        loss, acc = self.model.evaluate(self.data.images_testing_sample,self.data.labels_testing_sample)
        print("loss = " , loss , " || acc = ", acc)
        
        return (loss,acc)
    
    def evaluate(self):
        
        loss, acc = self.model.evaluate(self.data.images_testing,self.data.labels_testing)
        print("loss = " , loss , " || acc = ", acc)
        
        return (loss,acc)
        
    def save_model(self, path_model):
        self.model.save(path_model)
        
    def load_model(self, path_model):
        self.model = load_model(path_model)
        
    def model_summary(self):
        self.model.summary()
        
        
        