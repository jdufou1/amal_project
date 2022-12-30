"""
Main algorithm of the paper COMMUTE-GAN: COMPTETITIVE MULTIPLE EFFICIENT GENERATIVE ADVERSARIAL NETWORKS
src : https://openreview.net/pdf?id=u_-XxuTcnJ7
Implemented by :
jeremy dufourmantelle 
santhos arichandra
ethan abitbol
"""

from Generator import Generator
from Discriminator import Discriminator
from classifier import Classifier

import random
import torch 
import torch.nn as nn
import numpy as np
import time

torch.autograd.set_detect_anomaly(True)

from torch.utils.tensorboard import SummaryWriter

class StateCOGAN(object) :
    
    def __init__(
            self,
            classifier : Classifier,
            discriminator : Discriminator,
            beta_1 : float,
            beta_2 : float,
            lambda_gp : float,
            n_d : int,
            gamma : float,
            nb_generator,
            batch_size : int,
            nb_epochs : int,
            current_epoch : int,
            learning_rate_generator : float,
            learning_rate_discriminator : float,
            save_frequency : int,
            save_path : str,
            device : str
        ) :
        
        self.classifier = classifier.to(device)
        self.discriminator = discriminator.to(device)
        self.list_generator = [Generator().to(device) for _ in range(nb_generator)]
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lambda_gp = lambda_gp
        self.n_d = n_d
        self.gamma = gamma
        self.nb_generator = nb_generator
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.current_epoch = current_epoch
        self.learning_rate_generator = learning_rate_generator
        self.learning_rate_discriminator = learning_rate_discriminator
        
        self.optimizer_generator = [torch.optim.Adam(self.list_generator[i].parameters(), lr=learning_rate_generator , betas=(beta_1 , beta_2)) for i in range(nb_generator)]
        self.optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_discriminator , betas=(beta_1 , beta_2))
        
        self.save_frequency = save_frequency
        self.save_path = save_path
        self.device = device  

def tvd_loss(P , Q):
  return 0.5 * (P - Q).abs().sum(axis = 1)   
        
        
class COGANTraining : 
    
    def __init__(
        self,
        state : StateCOGAN
        ) :
        
        super(COGANTraining, self).__init__()
        
        self.state = state
        
        print("COGAN info :")
        print("batch_size : ",self.state.batch_size)
        print("nb_epochs : ",self.state.nb_epochs)
        print("current_epoch : ",self.state.current_epoch)
        print("save_frequency : ",self.state.save_frequency)
        print("save_path : ",self.state.save_path)
        print("device : ",self.state.device)
        print("Beta 1 : ", self.state.beta_1) 
        print("Beta 2 : ", self.state.beta_2) 
        print("Beta Lambda GP : ", self.state.lambda_gp) 
        print("Nb Generators : ", self.state.nb_generator)
        print("LR Generator : ", self.state.learning_rate_generator)
        print("LR Discriminort : ", self.state.learning_rate_discriminator)
        print("Gamma : ", self.state.gamma)
        print("Nb dicriminator iteration for 1 gen iteration : ", self.state.n_d)
        
        self.writer = SummaryWriter("./logs/amal_project_cogan")
        
    
    def training(self , data) :
        
        #data_loader = torch.utils.data.DataLoader(data,batch_size=self.batch_size,shuffle=True)
        #data_l = list(data)
        
        start_time = time.time()
        
        for epoch in range(self.state.current_epoch , self.state.nb_epochs): 
            
            self.state.current_epoch = epoch
            
            discriminator_loss_list = list()
            
            for _ in range(self.state.n_d) : 
                
                data_list = list(data)

                # line 3 :
                data_sampled = random.sample(data_list , k = self.state.batch_size)
                x = torch.stack([item[0][0] for item in data_sampled]).unsqueeze(-1).transpose(1,3).to(self.state.device)
                z = torch.randn(self.state.nb_generator , self.state.batch_size // self.state.nb_generator , 128).to(self.state.device)
                eps = random.random() # uniform sampling in [0-1]
                
                # line 5 & 6:
                x_b = torch.stack([self.state.list_generator[i](z[i]) for i in range(self.state.nb_generator)]).reshape(self.state.batch_size, 1 , 32 , 32)

                # line 7 : 
                x_hat = eps * x + (1 - eps) * x_b
                
                # line 8 :
                gradient_x_hat = torch.autograd.grad(outputs=self.state.discriminator(x_hat).mean(),inputs=x_hat)[0]      
                discriminator_loss = (self.state.discriminator(x_b) - self.state.discriminator(x) + self.state.lambda_gp * (torch.norm(gradient_x_hat) - 1)**2).mean()
                
                # line 9 :
                self.state.optimizer_discriminator.zero_grad()
                discriminator_loss.backward(retain_graph = True)
                self.state.optimizer_discriminator.step()
                
                discriminator_loss_list.append(discriminator_loss.item())
                
            
            # line 11 :
            z = torch.randn(self.state.nb_generator , self.state.batch_size , 128).to(self.state.device)

            # line 12 :
            # x_b = torch.stack([self.state.list_generator[i](z[i].detach()) for i in range(self.state.nb_generator)]) 
            x_b = [self.state.list_generator[i](z[i].detach()) for i in range(self.state.nb_generator)]
            # print(x_b.shape)
            
            generator_loss_list = list()

            # line 13 :
            if self.state.gamma != 0 :
                delta = 0.0
                for i in range(self.state.nb_generator) :
                    for j in range(i + 1 , self.state.nb_generator) :
                        delta += tvd_loss( self.state.classifier(x_b[i]) , self.state.classifier(x_b[j]) )
                delta /= self.state.nb_generator

                for i in range(self.state.nb_generator) :
                    loss_generator = (-self.state.discriminator(x_b[i]) + self.state.gamma*(1 - delta)).mean()
                    self.state.optimizer_generator[i].zero_grad()
                    loss_generator.backward(retain_graph = True)
                    self.state.optimizer_generator[i].step()
                    generator_loss_list.append(loss_generator.item())
            # line 16 :
            else :
                # line 19 :
                for i in range(self.state.nb_generator) :
                    loss_generator = -self.state.discriminator(x_b[i]).mean()
                    self.state.optimizer_generator[i].zero_grad()
                    loss_generator.backward(retain_graph = True)
                    self.state.optimizer_generator[i].step()
                    generator_loss_list.append(loss_generator.item())
                    
            generator_loss_list = np.array(generator_loss_list)
            discriminator_loss_list = np.array(discriminator_loss_list)
            
            for i in range(self.state.nb_generator) :
                name = "loss generator "+str(i)
                self.writer.add_scalar(name, generator_loss_list[i], self.state.current_epoch)
            self.writer.add_scalar('loss discriminator', generator_loss_list.mean(), self.state.current_epoch)
            
            if epoch % self.state.save_frequency == 0 :
                self.save_model()
                
                end_time = time.time()
                total_time = end_time - start_time
                start_time = time.time()
                
                print(f"[LOG] : {self.state.current_epoch}/{self.state.nb_epochs} - generators loss : {generator_loss_list.mean()} - discriminator loss : {discriminator_loss_list.mean()}- time : {round(total_time,3)}s")

                
    def save_model(self) :
        torch.save(self.state , self.state.save_path)
        print("[LOG COGAN] : save complete")            