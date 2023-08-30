
from src.models.mnist_cnn import Mnist_CNN

def choose_model(options):
    model_name = str(options['model_name']).lower()
    if model_name == 'mnist_cnn':
        return Mnist_CNN()



