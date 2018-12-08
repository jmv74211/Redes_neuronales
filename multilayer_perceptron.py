from neuronalNetwork import NeuronalNetwork
from keras.layers import Dense,Flatten, Dropout
from keras.optimizers import RMSprop
from tensorflow.python.keras.models import load_model

class MultilayerPerceptron(NeuronalNetwork):
    
    def __init__(self, data_source,epocs, neuronal_network_type):
        super(MultilayerPerceptron, self).__init__(data_source,epocs, neuronal_network_type)
    
     
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
        
    def save_model(self, num_epocs, num_dense, path=None):
        if path is None:
            self.model.save('./models/multilayer_perceptron/multilayer_perceptron_' + repr(num_dense) + 'n_' + repr(num_epocs) +'e.model')
        else:
            self.model.save(path)
            
    def load_model(self, num_epocs, num_dense, path=None):
        if path is None:
            self.model = load_model('./models/multilayer_perceptron/multilayer_perceptron_' + repr(num_dense) + 'n_' + repr(num_epocs) +'e.model')
        else:
            self.model = load_model(path)
    