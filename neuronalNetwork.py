from data import Dataset
import tensorflow as tf

class NeuronalNetwork:
    
    def __init__(self, data_source,_epocs):
        
        # Número de épocas
        self.epocs = _epocs
        
        # Cargo los datos
        self.data = Dataset(data_source)
        
        # Creo el modelo
        self.model = tf.keras.models.Sequential()
        
        
    
    def layers_construction(self):
        
        # Aplana la matriz de datos (números)
        self.model.add(tf.keras.layers.Flatten())
        
        # 128 neuronas, función activación
        self.model.add(tf.keras.layers.Dense(128,tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(128,tf.nn.relu))
        
        # Capa de salida
        self.model.add(tf.keras.layers.Dense(10,tf.nn.softmax))
                       
    def learn(self):
        
        self.model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
   
        self.model.fit(self.data.images_training,self.data.labels_training, epochs=self.epocs)
        
        