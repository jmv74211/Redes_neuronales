from mnist import MNIST
from number import Number


"""
 Clase que representa al conjunto de datos
"""

class Dataset:
    
    def __init__(self, data_source):
        self.mndata = MNIST(data_source)
        
        # Training data
        imagesTraining = self.mndata.load_training()[0]
        labelsTraining = self.mndata.load_training()[1]
        
        # Testing data
        imagesTesting = self.mndata.load_testing()[0]
        labelsTesting = self.mndata.load_testing()[1]
        
        #Cargo las imágenes como números
        self.numbersTraining = self.load_numbers_training(imagesTraining)
        self.numbersTesting = self.load_numbers_test(imagesTesting)
        
        
    def load_numbers_training(self,imagesTraining):
        
        numbersTraining = []
        
        for img in imagesTraining:
            numbersTraining.append( Number(img))
            
        return numbersTraining
            
    def load_numbers_test(self,imagesTest):
        
        numbersTesting = []
        
        for img in imagesTest:
            numbersTesting.append( Number(img))
        
        return numbersTesting
        
        
    
    