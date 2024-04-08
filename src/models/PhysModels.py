import torch
import torch.nn as nn

class Damped_oscillation(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.alpha = torch.tensor([-0.5], requires_grad=True).float()
        self.alpha = nn.Parameter(self.alpha )
        self.beta = torch.tensor([-0.5], requires_grad=True).float()
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
    
class Oscillation(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.alpha = torch.tensor([-0.5], requires_grad=True).float()
        self.alpha = nn.Parameter(self.alpha )

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z[:,1:2]
      y0 = z[:,0:1]

      dt = dt

      y_hat = y1+ (y1-y0) -dt*dt *self.alpha* y1


      return  y_hat
    
class Sprin_ode(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.k = torch.tensor([1.5], requires_grad=True).float()
        self.k = nn.Parameter(self.k )

        self.l = torch.tensor([1.5], requires_grad=True).float()
        self.l = nn.Parameter(self.l )

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      pos1_t0 = z[:,0:1, 0:1]
      pos1_t1 = z[:,1:2, 0:1]

      pos2_t0 = z[:,0:1, 1:2]
      pos2_t1 = z[:,1:2, 1:2]

      vel1_t0 = torch.abs(pos1_t1 - pos1_t0)/dt
      vel2_t0 = torch.abs(pos2_t1 - pos2_t0)/dt
      
      distance = torch.norm(pos1_t0 - pos2_t0, dim=0)

      y1 = z[:,1:2]
      y0 = z[:,0:1]

      dt = dt
      
      y_hat = y1+ (y1-y0) -dt*dt *self.alpha* y1


      return  y_hat