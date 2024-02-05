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






class Encoder(nn.Module):
    def __init__(self, initw = False):
        super().__init__()

        #Linear layers
        self.linear = nn.ModuleList()

        self.linear.append(nn.Linear(2500,1000))
        self.linear.append(nn.Linear(1000,500))
        self.linear.append(nn.Linear(500,1))
        
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
    
class FramInt(nn.Module):
    def __init__(self, initw = False):
        super().__init__()

        #Linear layers
        self.linear = nn.ModuleList()

        self.linear.append(nn.Linear(30,15))
        self.linear.append(nn.Linear(15,5))
        self.linear.append(nn.Linear(5,3))
        
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
    
      for i in range(len(self.linear) -1):
            x = self.linear[i](x)
            x = self.Tanh(x)    
      x = self.linear[-1](x)
      return x

class pModel(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.alpha = torch.tensor([0.1], requires_grad=True).float()
        self.beta = torch.tensor([0.0], requires_grad=True).float()
        self.alpha = nn.Parameter(self.alpha )
        self.beta= nn.Parameter(self.beta)

    def forward(self, z,dt):    

      #device = "cuda" if torch.cuda.is_available() else "cpu"
      #dt = torch.tensor([dt], requires_grad=False).float().to(device)


      #return x1+(x1-x0)+self.alpha*(x1-x0 )*dt*2 + (self.beta*x1 )*dt*dt*4
        y0 = z[:,0:1]
        y1 = z[:,1:2]
        y2 = z[:,2:3]


        return y2 + (y2-y1) - dt*dt*(self.alpha*y1)

class Decoder(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.l1 = nn.Linear(1,100 , bias=False)
        self.l2 = nn.Linear(100,1000, bias=False)
        self.l3 = nn.Linear(1000,2500, bias=False)

        self.uflat = nn.Unflatten(1, torch.Size([50,50]))
        self.relu= nn.ReLU()
        self.Softmax= nn.Softmax(dim=1)
        self.sigmoid= nn.Sigmoid()

        self.Tanh=nn.Tanh()

        if initw:

          #uniform_
          #xavier_normal
          nn.init.xavier_normal_(self.l1.weight)
          nn.init.xavier_normal_(self.l2.weight)
          #nn.init.xavier_normal_(self.l3.weight)

    def forward(self, x):

      x = self.relu(self.l1(x))
      x = self.relu(self.l2(x))
      x = self.l3(x)

      x = self.uflat(x)
      return x



class AE(nn.Module):
    def __init__(self, dt, initw = False):
        super().__init__()

        self.framencoder = Encoder(initw=initw)
        self.decoder     = Decoder(initw=initw)
        self.pModel      = pModel(initw=initw)
        self.dt = dt

    def forward(self, x0):
      
      batch, channel, height, width = x0.shape

      

      # iterate over the channels
     
      for i in range(channel):
         
        xframe = x0[:,i,:,:].unsqueeze(1)

        if False:
            max_values_list = []
            max_values_dim1, _ = torch.max(xframe, dim=1)
            max_values_dim2, _ = torch.max(max_values_dim1, dim=1)
            max_values_dim3, _ = torch.max(max_values_dim2, dim=1)
            max_values_dim3 = max_values_dim3.unsqueeze(1)
            max_values_list = torch.cat((max_values_list,max_values_dim3),1)
            max_values_list = max_values_dim
            print(max_values_list.shape, max_values_dim3.shape)
            print(max_values_list)

        xframe = self.framencoder(xframe)
        # concatenate the frames
        if i == 0:
            x = xframe
        else:
            x = torch.cat((x,xframe),1)    
            


      
      z = x
      z2 =self.pModel(z, self.dt)

      in0Rec =self.decoder(z[:,1:2])
      in1Rec =self.decoder(z[:,2:3])
      outRec =self.decoder(z2)

      return (z[:,0:1],z[:,2:3],z2), (in0Rec,in1Rec,outRec)