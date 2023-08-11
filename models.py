import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, initw = False):
        super().__init__()

        #Conv layers
        self.Conv = nn.Conv2d(16, 33, 3, stride=2)

        #Linear layers
        self.linear = nn.ModuleList()

        self.linear.append(nn.Linear(130000,500))
        self.linear.append(nn.Linear(500,10))
        self.linear.append(nn.Linear(10,1))
        
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
      x=torch.flatten(x,1)

      

      for i in range(len(self.linear) -1):
            x = self.linear[i](x)
            x = torch.relu(x)    
      x = self.linear[-1](x)

      return x

class pModel(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.alpha = torch.tensor([-2.00], requires_grad=True).float()
        self.beta = torch.tensor([-2500.00], requires_grad=True).float()
        self.alpha = nn.Parameter(self.alpha )
        self.beta= nn.Parameter(self.beta)

    def forward(self, x0,x1,dt):    

      #device = "cuda" if torch.cuda.is_available() else "cpu"
      #dt = torch.tensor([dt], requires_grad=False).float().to(device)


      return x1+(x1-x0)+self.alpha*(x1-x0 )*dt*2 + (self.beta*x1 )*dt*dt*4

class Decoder(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.l1 = nn.Linear(1,10 )
        self.l2 = nn.Linear(10,500 )
        self.l3 = nn.Linear(500,130000 )

        self.uflat = nn.Unflatten(1, torch.Size([1,260,500]))
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

        self.encoder     = Encoder(initw=initw )
        self.decoder     = Decoder(initw=initw)
        self.pModel     =  pModel(initw=initw)
        self.dt = dt

    def forward(self, x0,x1):

      z0 =self.encoder(x0)
      z1 =self.encoder(x1)
      z2 =self.pModel(z0,z1, self.dt)

      in0Rec =self.decoder(z0)
      in1Rec =self.decoder(z1)
      outRec =self.decoder(z2)

      return (z0,z1,z2), (in0Rec,in1Rec,outRec)