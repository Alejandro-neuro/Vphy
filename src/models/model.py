from . import unet

import torch
import torch.nn as nn
from . import blocks

class Encoder(nn.Module):
    def __init__(self, chan = [1, 3, 9], initw = False):
        super().__init__()

        self.unet = unet.build_unet()

        self.extractor = blocks.Extractor()
        
        

    def forward(self, x):
      
        masks = []
        
        for i in range(3):
            mask = self.unet(x[:,i:i+1,:,:])
            masks.append(mask)
        
        masks = torch.cat(masks, dim=1)

        print("masks:", masks.shape)
        
        v = []
        for i in range(3):

            d0 = x[:,i:i+1,:,:] * masks[:,i:i+1,:,:]
            v.append(self.extractor(d0))
        
        v = torch.cat(v, dim=1)

        print("v:", v.shape) 

        
        return v
class Decoder(nn.Module):
    def __init__(self, chan = [1, 3, 9], initw = False):
        super().__init__()


    def forward(self, x):


      return x
    
class pModel(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.alpha = torch.tensor([50.0], requires_grad=True).float()
        self.beta = torch.tensor([2500.0], requires_grad=True).float()
        self.alpha = nn.Parameter(self.alpha )
        self.beta= nn.Parameter(self.beta)

    def forward(self, z,dt):    

      #device = "cuda" if torch.cuda.is_available() else "cpu"
      #dt = torch.tensor([dt], requires_grad=False).float().to(device)


      #return x1+(x1-x0)+self.alpha*(x1-x0 )*dt*2 + (self.beta*x1 )*dt*dt*4

      return z[:,0:1]+ z[:,1:2]*dt -( self.alpha*z[:,1:2] + self.beta*z[:,2:3] )*dt*dt