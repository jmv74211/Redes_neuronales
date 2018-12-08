from neuronalNetwork import NeuronalNetwork
from tensorflow.python.keras.models import load_model
from keras.layers import Dense,Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D

class ConvolutionalNetwork(NeuronalNetwork):
    
    def __init__(self, data_source,epocs, neuronal_network_type):
        super(ConvolutionalNetwork, self).__init__(data_source, epocs, neuronal_network_type)
        
    def layers_construction(self, training_sample = False):
        
        if not training_sample:
            self.model.add(Convolution2D(32, 3, 3, input_shape=self.data.images_training[0].shape))
        else:
            self.model.add(Convolution2D(32, 3, 3, input_shape=self.data.images_training_sample[0].shape))
        
        self.model.add(Convolution2D(32, 3, 3, input_shape=self.data.images_training[0].shape,activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        
        self.model.add(Convolution2D(32,3,3,activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.2))
        
        self.model.add(Flatten())
        
        self.model.add(Dense(512,activation='relu'))
        self.model.add(Dropout(0.2))
        
        self.model.add(Dense(self.num_classes,activation='softmax'))
        
        self.model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
        
    def save_model(self, num_epocs, num_dense, path=None):
        if path is None:
            self.model.save('./models/convolutional_network/convolutional_network_' + repr(num_dense) + 'n_' + repr(num_epocs) +'e.model')
        else:
            self.model.save(path)
            
    def load_model(self, num_epocs, num_dense, path=None):
        if path is None:
            self.model = load_model('./models/convolutional_network/convolutional_network_' + repr(num_dense) + 'n_' + repr(num_epocs) +'e.model')
        else:
            self.model = load_model(path)
        