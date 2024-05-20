from . import unet

import torch
import torch.nn as nn
from . import blocks
from . import modelConv2d
from . import modelineal
from . import encoders
from . import PhysModels
from . import aunet

import numpy as np

import matplotlib.pyplot as plt

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

      dt = dt

      #for i in range(5):

      #y_hat = y1+ (y1-y0) -dt*dt *self.alpha* y1

      y_hat = y1 +(y1 - y0) - dt*(self.beta*(y1-y0) +dt*self.alpha*y1)

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
    def __init__(self, in_size=50, latent_dim = 1, n_mask = 1, in_channels = 1,  dt = 0.2, pmodel = "Damped_oscillation", init_phys = None, initw = False):
        super().__init__()

        self.n_mask = n_mask
        self.in_channels = in_channels 

        self.use_mask = False

        self.encoder = encoders.EncoderMLP(in_size = in_size, in_chan=in_channels, latent_dim = latent_dim, initw = False)
      
        self.masks = None
        #self.pModel = pModel()
        self.pModel = PhysModels.getModel(pmodel, init_phys)
        self.dt = dt
    def forward(self, x):    
      
      order = self.pModel.order
      frames = x.clone()

      #frame_area_list = [] 

     
      for i in range(frames.shape[1]):
          
          #frame = frames[batch,frame,channel,w,h]

          current_frame = frames[:,i,:,:,:]

          if self.use_mask:             
          
            mask = self.masker(current_frame) 
            mask_frame = current_frame * mask
            self.masks = mask if i == 0 else torch.cat((self.masks,mask),dim=1)
          
            z_temp = self.encoder(mask_frame)

          else:
            z_temp = self.encoder(frames[:,i,:,:,:])

          z_temp = z_temp.unsqueeze(1)
          z = z_temp if i == 0 else torch.cat((z,z_temp),dim=1)
    
      z2_phys = z[:,0:order,:]
      z2_encoder = z
      for i in range(frames.shape[1]-order):
          
          z_window = z[:,i:i+order]

          pred_window = self.pModel(z_window,self.dt)
                 
          z2_phys = torch.cat((z2_phys,pred_window),dim=1)
          
          #z2_phys = self.pModel(z[:,i:i+order],self.dt) if i == 0 else torch.cat((z2_phys,self.pModel(z[:,i:i+order],self.dt)),dim=1)


      

      
      return  z2_encoder, z2_phys
    
    def get_masks(self):
        return self.masks
    

class EndPhysMultiple(nn.Module):
    def __init__(self, in_size=50, latent_dim = 1, n_mask = 1, in_channels = 1,  dt = 0.2, pmodel = "Damped_oscillation", init_phys = None, initw = False):
        super().__init__()

        self.n_mask = n_mask
        self.in_channels = in_channels 

        self.latent_dim =latent_dim 

        self.m = nn.Parameter(torch.rand(1, requires_grad=True).float()+1.0 )
        self.b = nn.Parameter(torch.rand(1, requires_grad=True).float() )
        self.m = nn.Parameter(self.m )
        self.b = nn.Parameter(self.b )

        self.renorm = nn.Linear(4,4)
        
        self.encoder = encoders.EncoderMLP(in_size = in_size, in_chan=1, latent_dim = latent_dim, initw = False)
        self.encoder1 = encoders.EncoderMLP(in_size = in_size, in_chan=1, latent_dim = latent_dim, initw = initw)
        #self.masker = aunet.UNet(c_in = in_channels, c_out=n_mask, img_size=in_size)
        self.masks = None

        self.pModel = PhysModels.getModel(pmodel, init_phys)
        self.dt = dt
    def forward(self, x):    
      frames = x.clone()

      device = "cuda" if torch.cuda.is_available() else "cpu"   

      #frame_area_list = [] 


      for i in range(frames.shape[1]):
          
          #frame = frames[batch,frame,channel,w,h]

          #print(frames.shape)
          

          current_frame = frames[:,i,:,:,:]
          #print(current_frame.shape)
          if self.latent_dim == 2:
            mask1 = current_frame[:,0:1,:,:]
            mask2 = current_frame[:,1:2,:,:]
            p1 = self.encoder(mask1)
            p2 = self.encoder(mask2)            

            z_temp = torch.cat((p1.unsqueeze(1),p2.unsqueeze(1)),dim=2)

          if self.latent_dim == 4:
             mask1 = current_frame[:,0:1,:,:]
             mask2 = current_frame[:,1:2,:,:]
             z_temp = self.encoder(mask1+mask2 )
             z_temp = z_temp.unsqueeze(1)
          #print("z_temp",z_temp)
          z = z_temp if i == 0 else torch.cat((z,z_temp),dim=1)

         

      #print("z",z)
          
      z = z.squeeze(2)
      #print("z",z[0:5,0:5])
      m = torch.zeros(4).to(device)
      m[:] = self.m
      b = torch.zeros(4).to(device)
      b[2:] = self.b
      
      z_renorm = z#+b
      #z_renorm = self.renorm(z)
      #print("z_renorm",z_renorm[0:5,0:5])
      z_renorm = z[:,0:2,:]

      z2_phys = z_renorm[:,0:2,:]

      
      for i in range(frames.shape[1]-2):       
          

          z_window = z2_phys[:,i:i+2,:]
          z_window2 = z[:,i:i+2,:]


          pred_window = self.pModel(z_window,self.dt)
          pred_window2 = self.pModel(z_window2,self.dt)
          
          z2_phys = torch.cat((z2_phys,pred_window),dim=1)
          z_renorm = torch.cat((z_renorm,pred_window2),dim=1)

      #print(z2_phys.shape)
      #z2_encoder = z[:,2:]
      
      return  z, z2_phys, z_renorm
    
    def get_masks(self):
        return self.masks

        