import torch
import torchvision
from tqdm import tqdm

from torchvision import transforms
from classifier import ClassifierTraining,StateClassifier,Classifier

SAVE_PATH = "./classifier_model.pt"

state_classifier = torch.load(SAVE_PATH)

data_mnist = torchvision.datasets.MNIST(root='./data',transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]), download=True,train=True)

total_images = 0
good_pred = 0
for X,y in tqdm(data_mnist) :
    X = X.unsqueeze(0)
    X = X.to(state_classifier.device)
    total_images += X.shape[0]
    y_hat = state_classifier.classifier(X).cpu().detach().numpy()
    good_pred += (y == y_hat.argmax())


print(f"Accuracy classifier on train dataset : {round(good_pred/total_images*100,5)}%")
