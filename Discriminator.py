import torch
import torch.nn as nn

class Discriminator(nn.Module) : 
    
    def __init__(self) : 
        
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=(5,5),padding=(2,2),stride=(2,2)),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(64,128,kernel_size=(5,5),padding=(2,2),stride=(2,2)),
            nn.LayerNorm(8),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(128,256,kernel_size=(5,5),padding=(2,2),stride=(2,2)),
            nn.LayerNorm(4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv2d(256,512,kernel_size=(5,5),padding=(2,2),stride=(2,2)),
            nn.LayerNorm(2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(2048,1),
            nn.Sigmoid()
        )
        
    def forward(self,x) :
        return self.net(x)