from data import Dataset
from tensorflow.python.keras.models import load_model
from keras.layers import Dense,Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop


class MultilayerNetwork:
    
    def __init__(self, data_source,_epocs):
        
        # Número de épocas
        self.epocs = _epocs
        
        # Número de neuronas de salida
        self.num_classes = 10
        
        # Cargo los datos
        self.data = Dataset(data_source)
        
        # Creo el modelo
        self.model = Sequential()
     
    def layers_construction(self, training_sample = False):
        
        # Aplana la matriz de datos (números)
        if not training_sample:
            self.model.add(Flatten(input_shape=self.data.images_training[0].shape))
        else:
            self.model.add(Flatten(input_shape=self.data.images_training_sample[0].shape))
            
        # 512 neuronas, función activación
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        
        # Capa de salida
        self.model.add(Dense(self.num_classes, activation='softmax'))
        
        self.model.compile(optimizer = RMSprop() , loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
                       
    def learn_training_sample(self):
        
        self.model.fit(self.data.images_training_sample,self.data.labels_training_sample, epochs=self.epocs)
    
    def learn(self):
        
        self.model.fit(self.data.images_training,self.data.labels_training,batch_size=128,epochs=self.epocs)
    
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