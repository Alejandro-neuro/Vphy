import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def savekernels(a):

    batch, channel, height, width = a.shape

    b = torch.zeros(batch,1, height*channel, width)

    for i in range(3):
        b[:,:,i*height:(i+1)*height,:] = a[:,i,:,:].unsqueeze(1)
    
    imgs = torchvision.utils.make_grid(b)

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(nrows=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i, 0].imshow(np.asarray(img))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.savefig(f'./plots/modelconv.png')
    plt.close()

class ConvEncoder1(nn.Module):
    def __init__(self, chan = [1, 3, 9], initw = False):
        super().__init__()


        #convs layers
        self.convs = nn.ModuleList()

        for i in range(len(chan)):
          #self.convs.append(nn.Conv3d(chan[i], chan[i+1], kernel_size=3, stride=1, padding=0, bias=True))
          self.convs.append(nn.Conv3d(1, 3, kernel_size=3, stride=1, padding=0, bias=True))

        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3 ), stride=2)
        
        #Activation functions
        self.sigmoid = nn.Sigmoid()
        self.Tanh=nn.Tanh()
        self.relu= nn.ReLU()

        if initw:
          for m in self.modules():        
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):

      x = x.unsqueeze(1)

      for i in range(len(self.convs) -1):
            x = self.convs[i](x)
            x = torch.relu(x)
            x = self.maxpool(x)    
            x = x.permute(0,2,1,3,4) 

      x = self.convs[-1](x)
      x = x.permute(0,2,1,3,4)
      x = x.squeeze(1)

      savekernels(x)

      return x
    
class ConvEncoder(nn.Module):
    def __init__(self, chan = [1, 3, 9], initw = False):
        super().__init__()


        #convs layers
        self.convs = nn.ModuleList()

        self.convs.append(nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0, bias=True))

        for i in range(len(chan)):
          self.convs.append(nn.Conv2d(chan[i], chan[i+1], kernel_size=3, stride=1, padding=0, bias=True))
          

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3d = nn.MaxPool3d(kernel_size=(1,3,3 ), stride=2)
        
        #Activation functions
        self.sigmoid = nn.Sigmoid()
        self.Tanh=nn.Tanh()
        self.relu= nn.ReLU()

        if initw:
          for m in self.modules():        
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):

      x = x.unsqueeze(1)

      for i in range(len(self.convs) -1):
            x = self.convs[i](x)
            x = torch.relu(x)
            x = self.maxpool(x)    
            x = x.permute(0,2,1,3,4) 

      x = self.convs[-1](x)
      x = x.permute(0,2,1,3,4)
      x = x.squeeze(1)

      savekernels(x)

      return x
class ConvDecoder(nn.Module):
  def __init__(self, chan = [3, 2, 1], initw = False):
        super().__init__()


        #convs layers
        self.convs = nn.ModuleList()

        for i in range(len(chan)-1):
          self.convs.append(nn.ConvTranspose2d(chan[i], chan[i+1], kernel_size=3, stride=2, padding=0, bias=True))
          self.convs.append(nn.Conv2d(chan[i+1], chan[i+1], kernel_size=3, padding=4))

        self.convs.append(nn.Conv2d(1, 1, kernel_size=2, stride=1))


        
        #Activation functions
        self.sigmoid = nn.Sigmoid()
        self.Tanh=nn.Tanh()
        self.relu= nn.ReLU()

        if initw:
          for m in self.modules():        
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

  def forward(self, x):

    for i in range(len(self.convs) -1):
          x = self.convs[i](x)
          if i%2!=0:
            x = torch.relu(x) 

    x = self.convs[-1](x)

    x =x.squeeze(1)


    return x
class Encoder(nn.Module):
    def __init__(self, initw = False):
        super().__init__()

        #Linear layers
        self.linear = nn.ModuleList()

        self.linear.append(nn.Linear(21600,500))
        self.linear.append(nn.Linear(500,10))
        self.linear.append(nn.Linear(10,3))
        
        #Activation functions
        self.sigmoid = nn.Sigmoid()
        self.Tanh=nn.Tanh()
        self.relu= nn.ReLU()

        if initw:
          for m in self.modules():        
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):      
      x=torch.flatten(x,1)     

      for i in range(len(self.linear) -1):
            x = self.linear[i](x)
            x = torch.relu(x)    
      x = self.linear[-1](x)

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

class Decoder(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.l1 = nn.Linear(1,10 )
        self.l2 = nn.Linear(10,500 )
        self.l3 = nn.Linear(500,21600 )

        self.uflat = nn.Unflatten(1, torch.Size([3,60,120]))
        self.relu= nn.ReLU()
        self.Softmax= nn.Softmax(dim=1)
        self.sigmoid= nn.Sigmoid()

        self.Tanh=nn.Tanh()

        if initw:

          #uniform_
          #xavier_normal
          nn.init.xavier_normal_(self.l1.weight)
          nn.init.xavier_normal_(self.l2.weight)
          nn.init.xavier_normal_(self.l3.weight)

    def forward(self, x):


      x = self.Tanh(self.l1(x))
      x = self.Tanh(self.l2(x))
      x = self.sigmoid(self.l3(x))

      x = self.uflat(x)
      return x

class AE(nn.Module):
    def __init__(self, dt, initw = False):
        super().__init__()

        self.convencoder = ConvEncoder(initw=initw )
        self.encoder     = Encoder(initw=initw )
        self.decoder     = Decoder(initw=initw)
        self.convdecoder = ConvDecoder(initw=initw)  
        self.pModel      = pModel(initw=initw)
        self.dt = dt

    def forward(self, x0):

      

      xtemp = self.convencoder(x0)
      z =self.encoder(xtemp)
      z2 =self.pModel(z, self.dt)

      in0Rec =self.convdecoder(self.decoder(z[:,0:1]))
      in1Rec =self.convdecoder(self.decoder(z[:,2:3]))
      outRec =self.convdecoder(self.decoder(z2))

      return (z[:,0:1],z[:,2:3],z2), (in0Rec,in1Rec,outRec)