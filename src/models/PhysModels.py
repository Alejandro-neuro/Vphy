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
        self.k = torch.tensor([1.0], requires_grad=True).float()
        self.k = nn.Parameter(self.k )

        self.l = torch.tensor([10.0], requires_grad=True).float()
        self.l = nn.Parameter(self.l )

    def forward(self, z,dt):    
      #print("z",z.shape)
      
      #z = z.reshape(-1, 2, 2)

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      pos1_t0 = z[:,0,0:2]
      pos1_t1 = z[:,1,0:2]

      pos2_t0 = z[:,0,2:4]
      pos2_t1 = z[:,1,2:4]

      vel1_t1 = (pos1_t1 - pos1_t0)/dt
      vel2_t1 = (pos2_t1 - pos2_t0)/dt 

      #print("pos1_t0",pos1_t0)
      #print("pos2_t0",pos2_t0)
      
      norm = torch.norm(pos1_t0 - pos2_t0, dim=1, keepdim=True) 
      direction = (( pos1_t1 - pos2_t1 )/ (norm+1e-4)  )
      

      #if torch.isnan(direction).any():
       
        

      #force  = self.k*(norm - 2*self.l)*direction
      force = -(self.k*( pos1_t1 - pos2_t1 ) + self.l*direction)

      pos1_t2 = pos1_t1 + vel1_t1*dt + force*dt*dt
      pos2_t2 = pos2_t1 + vel2_t1*dt - force*dt*dt
      
      z_hat = torch.cat([pos1_t2 ,pos2_t2],dim=1).unsqueeze(1)

      return  z_hat
    
class gravity_ode(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.g = torch.tensor([1.0], requires_grad=True).float()
        self.g = nn.Parameter(self.g )

        

    def forward(self, z,dt):    
      #print("z",z.shape)
      
      #z = z.reshape(-1, 2, 2)

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      pos_t0 = z[:,0,:]
      pos_t1 = z[:,1,:]

      vel = (pos_t1 - pos_t0)/dt

      
      
      norm = torch.norm(pos1_t0 - pos2_t0, dim=1, keepdim=True) 
      direction = (( pos1_t1 - pos2_t1 )/ (norm+1e-4)  )
      

      #if torch.isnan(direction).any():
       
        

      #force  = self.k*(norm - 2*self.l)*direction
      force = -(self.k*( pos1_t1 - pos2_t1 ) + self.l*direction)

      pos1_t2 = pos1_t1 + vel1_t1*dt + force*dt*dt
      pos2_t2 = pos2_t1 + vel2_t1*dt - force*dt*dt
      
      z_hat = torch.cat([pos1_t2 ,pos2_t2],dim=1).unsqueeze(1)

      return  z_hat