from . import unet

import torch
import torch.nn as nn
from . import blocks
from . import modelConv2d
from . import modelineal
from . import encoders

class Encoder(nn.Module):
    def __init__(self, initw = False):
        super().__init__()

        self.unet = unet.build_unet()
        self.extractor = blocks.Extractor() 
        self.background = None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mask = torch.ones(100, 100, requires_grad=True).to(device)
        self.mask = nn.Parameter(self.mask )

    def forward(self, x):
      
        masks = []            
        masks = self.unet(x)
        mask = masks[:,0:1,:,:]
        self.background = masks[:,0:1,:,:]
        
        #masks = torch.cat(masks, dim=1)

       
        
        v = []
        for i in range(x.shape[1]):

            d0 = x[:,i:i+1,:,:] * mask
            
            vector = d0.view(d0.shape[0], -1)
            
            v.append(self.extractor(vector))
        
        v = torch.cat(v, dim=1)

        

      
        
        return v,mask
class Decoder(nn.Module):
    def __init__(self, chan = [1, 3, 9], initw = False):
        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base = torch.ones(100, 100, requires_grad=True).to(device)
        self.base = nn.Parameter(self.base )

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.conv_layer2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.relu= nn.ReLU()

        self.temp = None
        


    def forward(self, v,mask):
      
      x = v.view(v.shape[0], 1, 1, 1) #* mask
      x = x + self.base
      self.temp = x
      x = self.relu(self.conv_layer1(x))
      x = self.relu(self.conv_layer2(x))   

      return x
    
class pModel(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.alpha = torch.tensor([1.0], requires_grad=True).float()
        self.alpha = nn.Parameter(self.alpha )
        self.beta = torch.tensor([1.0], requires_grad=True).float()
        self.beta = nn.Parameter(self.beta )

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z[:,1:2]
      y0 = z[:,0:1]

      dt = dt/5

      for i in range(5):

        y_hat = y1+ (y1-y0) -dt*dt *self.alpha* y1

        y0 = y1
        y1 = y_hat


      #return x1+(x1-x0)+self.alpha*(x1-x0 )*dt*2 + (self.beta*x1 )*dt*dt*4

      return  y_hat
 
class AEModel(nn.Module):
    def __init__(self, dt = 0.1, initw = False):
        super().__init__()
        self.encoder = Encoder()
        #self.decoder = Decoder()
        self.decoder = modelineal.Decoder()
        self.pModel = pModel()

        self.dt = dt

        self.mask = None
        

    def forward(self, x):    
      x = x[:,1:3,:,:]
      v,mask = self.encoder(x)
      v_pred = self.pModel(v,self.dt)

      in0Rec = self.decoder(v[:,0:1])
      in1Rec = self.decoder(v[:,1:2] )
      outRec = self.decoder(v_pred)

      self.mask = mask

      return  (v[:,0:1],v[:,1:2],v_pred), (in0Rec,in1Rec,outRec)
    

class EndPhys(nn.Module):
    def __init__(self, dt = 0.2, initw = False):
        super().__init__()
        self.encoder = encoders.EncoderMLP()
        self.pModel = pModel()

        self.dt = dt
    def forward(self, x):    
      frames = x.clone()

      for i in range(frames.shape[1]):
          
          z_temp = self.encoder(frames[:,i:i+1,:,:]) 
          z = z_temp if i == 0 else torch.cat((z,z_temp),dim=1)

      for i in range(frames.shape[1]-2):
          
          z2_phys = self.pModel(z[:,i:i+2],self.dt) if i == 0 else torch.cat((z2_phys,self.pModel(z[:,i:i+2],self.dt)),dim=1)


      z2_encoder = z[:,2:]

      
      return  z2_encoder, z2_phys

        