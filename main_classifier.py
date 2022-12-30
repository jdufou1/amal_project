import torch
import torchvision
import numpy as np
import sys

from torchvision import transforms
from classifier import ClassifierTraining,StateClassifier,Classifier

SAVE_PATH = "./classifier_model.pt"

if __name__ == '__main__':
    
    args = sys.argv[1:]
    
    restart = len(args) > 0
    
    if restart :
        state_classifier = torch.load(SAVE_PATH)
    else :
        device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = Classifier()
        state_classifier = StateClassifier(
            classifier = classifier,
            batch_size = 64,
            nb_epochs = 100000,
            current_epoch = 0,
            learning_rate = 1e-3,
            save_frequency = 50,
            save_path = SAVE_PATH,
            device = device
        )
        
    clf_training = ClassifierTraining(state_classifier)
    
    data_mnist = torchvision.datasets.MNIST(root='./data',transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ]), download=True,train=True)
    
    clf_training.training(data_mnist)