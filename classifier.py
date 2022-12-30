"""
Classifier of the paper COMMUTE-GAN: COMPTETITIVE MULTIPLE EFFICIENT GENERATIVE ADVERSARIAL NETWORKS
src : https://openreview.net/pdf?id=u_-XxuTcnJ7
Implemented by :
jeremy dufourmantelle 
santhos arichandra
ethan abitbol
"""

import torch
import torch.nn as nn
import numpy as np
import time

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Classifier(nn.Module):
    def __init__(self, num_classes = 10):
        super(Classifier, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride = 2,padding=(2,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(64, 128, kernel_size=5,  stride = 2,padding=(2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(128, 256, kernel_size=5,  stride = 2,padding=(2,2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(256, 512, kernel_size=5,  stride = 2,padding=(2,2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.mlp_block = nn.Sequential(
            nn.Linear(2048, 128),
            nn.Linear(128, 128),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = nn.Dropout(p=0.3)(x)
        x = self.mlp_block(x)
        return x

class StateClassifier(object) :
    
    def __init__(
            self,
            classifier,
            batch_size,
            nb_epochs,
            current_epoch,
            learning_rate,
            save_frequency,
            save_path,
            device
        ) :
        
        self.classifier = classifier.to(device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.current_epoch = current_epoch
        self.learning_rate = learning_rate
        self.save_frequency = save_frequency
        self.save_path = save_path
        self.device = device        
    
class ClassifierTraining :
    
    def __init__(
            self,
            state : StateClassifier
                 ) -> None:
        super(ClassifierTraining, self).__init__()
        
        self.state = state
        
        print("Classifier info :")
        print("optimizer : ",self.state.optimizer)
        print("batch_size : ",self.state.batch_size)
        print("nb_epochs : ",self.state.nb_epochs)
        print("current_epoch : ",self.state.current_epoch)
        print("learning_rate : ",self.state.learning_rate)
        print("save_frequency : ",self.state.save_frequency)
        print("save_path : ",self.state.save_path)
        print("device : ",self.state.device)
        
        self.writer = SummaryWriter("./logs/amal_project_classifier")
        
    def training(self,data) :
        
        data_loader = torch.utils.data.DataLoader(data,batch_size=64,shuffle=True)
        data_l = list(data)
        
        start_time = time.time()
        
        for epoch in range(self.state.current_epoch, self.state.nb_epochs) :
            
            self.state.current_epoch = epoch
            
            losses = list()
            
            for X,y in data_loader : 
                
                X = X.to(self.state.device)
                y = y.to(self.state.device)
                
                y_hat = self.state.classifier(X)
                
                ce_loss = nn.CrossEntropyLoss()

                loss = ce_loss(input = y_hat, target = y)
                
                self.state.optimizer.zero_grad()
                loss.backward()
                self.state.optimizer.step()

                losses.append(loss.item())
        
            losses = np.array(losses)
            
            self.writer.add_scalar('loss', losses.mean(), self.state.current_epoch)
            
            if epoch % self.state.save_frequency == 0 :
                self.save_model()
                score = self.test_training(data_loader,len(data_l))
                
                end_time = time.time()
                total_time = end_time - start_time
                start_time = time.time()
                print(f"[LOG] : {self.state.current_epoch}/{self.state.nb_epochs} - train acc : {score} - train loss : {losses.mean()} - time : {round(total_time,3)}s")

    def test_training(self,data_loader,size) : 
        score = 0.0
        for X,y in data_loader :
            y_hat = self.state.classifier(X).argmax(axis=1)
            results = (y_hat == y)
            score += results.sum() 
        score /= size
        self.writer.add_scalar('train score', score, self.state.current_epoch)
        return score

    def save_model(self) :
        torch.save(self.state , self.state.save_path)
        print("[LOG CLASSIFIER] : save complete")