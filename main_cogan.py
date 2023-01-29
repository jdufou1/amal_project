import torch
import torchvision
import numpy as np
import sys

from torchvision import transforms
from classifier import ClassifierTraining,StateClassifier,Classifier
from CoGan import COGANTraining,StateCOGAN
from Discriminator import Discriminator
from Generator import Generator

CLASSIFIER_MODEL_PATH = "./classifier_model.pt"

SAVE_PATH = "./cogan_model.pt"

if __name__ == '__main__':
    
    args = sys.argv[1:]
    
    restart = len(args) > 0
    
    if restart :
        state_cogan = torch.load(SAVE_PATH)
    else :
        device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_classifier = torch.load(CLASSIFIER_MODEL_PATH)
        
        discriminator = Discriminator()
        generator = Generator()
        
        """
        HPs of the paper

        state_cogan = StateCOGAN(
            classifier = state_classifier.classifier,
            discriminator = discriminator,
            beta_1 = 0.5,
            beta_2 = 0.9,
            lambda_gp = 10,
            n_d = 3,
            gamma = 12.5,
            nb_generator = 5,
            batch_size = 500,
            nb_epochs = 70,
            current_epoch = 0,
            learning_rate_generator = 0.0002,
            learning_rate_discriminator = 0.0002,
            save_frequency = 1,
            save_path = SAVE_PATH,
            device = device
        )
        """

        state_cogan = StateCOGAN(
            classifier = state_classifier.classifier,
            discriminator = discriminator,
            beta_1 = 0.5,
            beta_2 = 0.9,
            lambda_gp = 10,
            n_d = 3,
            gamma = 0,
            nb_generator = 2,
            batch_size = 500,
            nb_epochs = 70,
            current_epoch = 0,
            learning_rate_generator = 0.0002,
            learning_rate_discriminator = 0.0002,
            save_frequency = 1,
            save_path = SAVE_PATH,
            device = device
        )


        
    cogan_training = COGANTraining(state_cogan)
    
    data_mnist = torchvision.datasets.MNIST(root='./data',transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((.5), (.5))
            ]), download=True,train=True)
    



    cogan_training.training(data_mnist)