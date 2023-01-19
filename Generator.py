import torch 
import torch.nn as nn


#Intermediaire Layer

class Squeeze(nn.Module) : 
    def __init__(self) :
        super().__init__()
    def forward(self,x) :
        return x.squeeze(1)

class UnSqueeze(nn.Module) : 
    def __init__(self) :
        super().__init__()
    def forward(self,x) :
        return x.unsqueeze(1)

#### Generator model ####

class Generator(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    """ Input size 128
        Dense layer 128 -> 4096
        Batch Normalization
        LeakyRelu
    """
    
    self.model = nn.Sequential(
      nn.Linear(128, 4096, bias=False), # Gen inp
      nn.LeakyReLU(.2), #leaky_re_lu_2
      nn.BatchNorm1d(4096), #Batch normalization
      nn.Unflatten(1, (1,256,4,4)), #reshape

      #Unsample_block
      nn.Upsample(size=(128,8,8)),
      Squeeze(),
      nn.Conv2d(128, 128, (3,3), stride=1, padding=1, bias=False),
      nn.LeakyReLU(.2),
      nn.BatchNorm2d(128),
      UnSqueeze(),

      # Unsample_block 1
      nn.Upsample(size=(64,16,16)),
      Squeeze(),
      nn.Conv2d(64, 64, (3,3), stride=1, padding=1, bias=False),
      nn.LeakyReLU(.2),
      nn.BatchNorm2d(64),
      UnSqueeze(),

      # Unsample_block 2
      nn.Upsample(size=(1,32,32)),
      Squeeze(),
      nn.Conv2d(1, 1, (3,3), stride=1, padding=1, bias=False),
      nn.Tanh(),
      nn.BatchNorm2d(1),
      nn.Dropout(p=0.3),
    )

  def forward(self, x):
    return self.model(x)