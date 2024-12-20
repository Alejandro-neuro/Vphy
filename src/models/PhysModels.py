import torch
import torch.nn as nn

class Damped_oscillation(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([init_phys*0.1], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.5], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        self.beta = nn.Parameter(self.beta )

        self.order = 2

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z[:,1:2]
      y0 = z[:,0:1]

      dt = dt

      #for i in range(5):

      #y_hat = y1+ (y1-y0) -dt*dt *self.alpha* y1

      #y_hat = y1 +(y1 - y0) - dt*(self.beta*(y1-y0) +dt*(9.80665/self.alpha)*torch.sin(y1) )

      y_hat = y1 +(y1 - y0) - dt*(self.beta*(y1-y0) +dt*self.alpha*y1 )

      return  y_hat
    
class dyn_1storder(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([init_phys], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.5], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        #self.beta = nn.Parameter(self.beta )

        self.order = 1

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z

      dt = dt

      y_hat= y1 - dt*self.alpha*y1

      return  y_hat
    
class lineal(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([init_phys], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.5], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        self.beta = nn.Parameter(self.beta )

        self.order = 1

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z

      dt = dt

      y_hat= y1 - dt*self.alpha

      return  y_hat
class Clifford_Attractor(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        self.a = torch.tensor([1.0], requires_grad=True).float()
        self.a = nn.Parameter(self.a )

        self.b = torch.tensor([1.0], requires_grad=True).float()
        self.b = nn.Parameter(self.b )

        self.c = torch.tensor([1.0], requires_grad=True).float()
        self.c = nn.Parameter(self.c )

        self.d = torch.tensor([1.0], requires_grad=True).float()
        self.d = nn.Parameter(self.d )

        self.order = 1

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"

      x = z[:,0,0:1]
      y = z[:,0,1:2]


      for i in range(10):

        x_hat = torch.sin(self.a*y) + 0.6*torch.cos(self.a*x)
        y_hat = torch.sin(self.b*x) + 1.2*torch.cos(self.b*y)

        x = x_hat
        y = y_hat
    
      z_hat = torch.cat([x_hat,y_hat],dim=1).unsqueeze(1)

      return  z_hat
    
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
    
class IntegratedFire(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.R = torch.tensor([-0.5], requires_grad=True).float()
        self.R = nn.Parameter(self.R )

        self.tau = torch.tensor([0.5], requires_grad=True).float()
        self.tau = nn.Parameter(self.tau )

        self.Vrest = torch.tensor([0.5], requires_grad=True).float()
        self.Vrest = nn.Parameter(self.Vrest )
        

        self.order = 1
        

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      v = z[:,0:1,0]
      i = z[:,0:1,1]
      

      dt = dt

      v_hat = v+dt * self.tau * ( self.Vrest - v + self.R*i)
      i_hat = i

      v_hat = torch.cat([v_hat,i_hat],dim=1).unsqueeze(1)
     
      return  v_hat

class Sprin_ode(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.k = torch.tensor([1.0], requires_grad=True).float()
        self.k = nn.Parameter(self.k )

        self.l = torch.tensor([1.0], requires_grad=True).float()
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
        self.g = torch.tensor([100.0], requires_grad=True).float()
        self.g = nn.Parameter(self.g )        

    def forward(self, z,dt):    
    
      
      #z = z.reshape(-1, 2, 2)

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      batch_size = z.shape[0]

      pos_t0 = z[:,0,:]
      pos_t1 = z[:,1,:]
      vel = (pos_t1 - pos_t0)

      pos_t1 = pos_t1.view(batch_size, -1, 2)
      vel = vel.view(batch_size, -1, 2)   

      distance = torch.cdist(pos_t1, pos_t1.roll(-1,1)).diagonal(dim1=1, dim2=2)
      distance = distance # guarantee no division by zero
      distance = distance.pow(3) # cube the distance
      distance = (1/distance) +1e-3  # inverse the distance
      distance = distance.unsqueeze(2).repeat(1,1,2) # repeat the distance to match the shape of the direction

      direction = (pos_t1 - pos_t1.roll(-1,1))

      forces = -self.g * torch.mul( direction , distance )
      forces = forces - forces.roll(1,1)

      z_hat = pos_t1 + vel + forces*dt*dt    

      #check Nan values
      if torch.isnan(z_hat).any():
        print("Nan values in gravity_ode")

      return  z_hat.view(-1,6).unsqueeze(1)
    

class ODE_2ObjectsSpring(nn.Module):
    def __init__(self, k, eq_distance):
        super().__init__()        
               
        self.eq_distance = torch.tensor([eq_distance], requires_grad=True).float()
        self.eq_distance = nn.Parameter(self.eq_distance)

        
        self.k = torch.tensor([k], requires_grad=True).float()
        self.k = nn.Parameter(self.k)

        self.relu = nn.ReLU()

    def force_eq(self, p1, p2):
        diff = (p2 - p1)
        euclidean_distance = torch.norm(diff, dim=1, keepdim=True)
        direction = (p2 - p1)/euclidean_distance
        #Force = self.k*(euclidean_distance - self.eq_distance )*direction
        #Force = torch.exp(self.k)*diff - torch.exp(self.k)*torch.exp(self.eq_distance)*direction
        Force = self.k*(euclidean_distance - 2*torch.abs(self.eq_distance) )*direction
        return Force
    def vel_eq(self, v):
        
        return v
    
    def runge_kutta_force(self,f, p1, p2, dt):
        k1 = f(p1, p2)
        k2 = f(p1 + dt/2, p2 + dt/2)
        k3 = f(p1 + dt/2, p2 + dt/2)
        k4 = f(p1 + dt, p2 + dt)
        return (k1 + 2*k2 + 2*k3 + k4)/6
    def runge_kutta_vel(self,f, v, dt):
        k1 = f(v)
        k2 = f(v + dt/2)
        k3 = f(v + dt/2)
        k4 = f(v + dt)
        return (k1 + 2*k2 + 2*k3 + k4)/6

    def forward(self, x, dt):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.tensor([dt], requires_grad=False).float().to(device)
    
        
        p1 = x[:,1,0:2]
        p2 = x[:,1,2:4]

        p1_0 = x[:,0,0:2]
        p2_0 = x[:,0,2:4]

        v1 = (p1 - p1_0)/dt
        v2 = (p2 - p2_0)/dt
            

        diff = (p1 - p2)

        
        #Force = self.runge_kutta_force(self.force_eq, p1, p2, dt)

        Force = self.force_eq(p1, p2)
        p1_new = 2*p1 -p1_0 + Force*dt*dt
        p2_new = 2*p2 -p2_0 - Force*dt*dt

        #Force = self.runge_kutta_force(self.force_eq, p1, p2, dt)

        #v11 = 
        #p1_new = p1 + self.runge_kutta_vel(self.vel_eq, v1 + Force , dt)
        #p2_new = p2 + self.runge_kutta_vel(self.vel_eq, v2 - Force , dt)



        

            #p1_0 = p1
            #p2_0 = p2
            #p1 = p1_new
            #p2 = p2_new

        
        z_hat = torch.cat((p1_new.unsqueeze(1) ,p2_new.unsqueeze(1)),dim=2)

        #print("z",x)

        #print("z_hat",z_hat)

        return z_hat
    
class double_pendulum(nn.Module):
    def __init__(self, initw = False):
        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.g = torch.tensor([9.8], requires_grad=True).float().to(device)
        #self.g = nn.Parameter(self.g)
        

        self.l1 = torch.tensor([1.0], requires_grad=True).float().to(device)
        #self.l1 = nn.Parameter(self.l1)
        

        self.l2 = torch.tensor([1.0], requires_grad=True).float().to(device)
        #self.l2 = nn.Parameter(self.l2)
        

        self.m1 = torch.tensor([1.0], requires_grad=True).float().to(device)
        #self.m1 = nn.Parameter(self.m1)
        

        self.m2 = torch.tensor([1.0], requires_grad=True).float().to(device)
        #self.m2 = nn.Parameter(self.m2)

        self.relu = nn.ReLU()

    #Get theta1 acceleration 
    def theta1_acceleration(self, theta1,theta2,theta1_velocity,theta2_velocity):

        g = self.g
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2
        mass1 = -g*(2*m1 + m2)*torch.sin(theta1)
        mass2 = -m2*g*torch.sin(theta1 - 2*theta2)
        interaction = -2*torch.sin(theta1 - theta2)*m2*torch.cos(theta2_velocity**2*l2 + theta1_velocity**2*l1*torch.cos(theta1 - theta2))
        normalization = l1*(2*m1 + m2 - m2*torch.cos(2*theta1 - 2*theta2))
        
        theta1_ddot = (mass1 + mass2 + interaction)/normalization
        
        return theta1_ddot

    #Get theta2 acceleration
    def theta2_acceleration(self, theta1,theta2,theta1_velocity,theta2_velocity):
        g = self.g
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        l2 = self.l2


        system = 2*torch.sin(theta1 - theta2)*(theta1_velocity**2*l1*(m1 + m2) + g*(m1 + m2)*torch.cos(theta1) + theta2_velocity**2*l2*m2*torch.cos(theta1 - theta2))
        normalization = l1*(2*m1 + m2 - m2*torch.cos(2*theta1 - 2*theta2))
        
        theta2_ddot = system/normalization
        return theta2_ddot

    def force_eq(self, dt, theta1, theta2, omega1, omega2):

        theta1_new = theta1 + omega1*dt + self.theta1_acceleration(theta1,theta2,omega1,omega2)*dt*dt
        theta2_new = theta2 + omega2*dt + self.theta2_acceleration(theta1,theta2,omega1,omega2)*dt*dt

        return theta1_new, theta2_new

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)
      
    #   theta1_0 = torch.asin(z[:,0,0] / self.l1)
    #   theta2_0 = torch.asin((z[:,0,2] - z[:,0,0] ) / self.l2)

    #   theta1 = torch.asin(z[:,1,0] / self.l1)
    #   theta2 = torch.asin((z[:,1,2] - z[:,1,0] ) / self.l2)

      
      theta1_0 = z[:,0,0]
      theta2_0 = z[:,0,1]

      theta1 = z[:,1,0]
      theta2 = z[:,1,1]
      
      omega1 = theta1 - theta1_0
      omega2 = theta2 - theta2_0

      theta1_new, theta2_new = self.force_eq(dt, theta1, theta2, omega1, omega2)

    #   x1 = self.l1*torch.sin(theta1_new) #Pendulum 1 x
    #   y1 = -self.l1*torch.cos(theta1_new) #Pendulum 1 y

    #   x2 = self.l1*torch.sin(theta1_new) + self.l2*torch.sin(theta2_new) #Pendulum 2 x
    #   y2 = -self.l1*torch.cos(theta1_new) - self.l2*torch.cos(theta2_new) #Pendulum 2 y

    #   z_hat = torch.cat([x1.unsqueeze(1),y1.unsqueeze(1),x2.unsqueeze(1),y2.unsqueeze(1)],dim=1)

      
      z_hat = torch.cat((theta1_new.unsqueeze(1) ,theta2_new.unsqueeze(1)),dim=1)
      z_hat = z_hat.unsqueeze(1)

      return  z_hat
    
class Pendulum(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([init_phys*0.1], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.5], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        self.beta = nn.Parameter(self.beta )

        self.order = 2

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z[:,1:2]
      y0 = z[:,0:1]

      dt = dt

      y_hat = y1 +(y1 - y0) - dt*(self.beta*(y1-y0) +dt*(9.80665/self.alpha)*torch.sin(y1) )

      return  y_hat

class Sliding_block(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([init_phys*0.1], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.5], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        self.beta = nn.Parameter(self.beta )

        self.order = 2

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z[:,1:2]
      y0 = z[:,0:1]

      dt = dt

      y_hat = y1 +(y1 - y0) - dt*dt*9.80665( torch.sin(self.alpha) - self.beta*torch.cos(self.alpha)   )       

      return  y_hat

class bouncing_ball(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([init_phys*0.1], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.5], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        self.beta = nn.Parameter(self.beta )

        self.order = 2

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z[:,1:2]
      y0 = z[:,0:1]

      dt = dt

      y_hat = y1 +(y1 - y0) - dt*dt*self.alpha        

      return  y_hat  
class led(nn.Module):
    def __init__(self, init_phys = None):
        super().__init__()

        if init_phys is not None:
            self.alpha = torch.tensor([init_phys], requires_grad=True).float()
            self.beta = torch.tensor([0.0], requires_grad=True).float()
        else:
            self.alpha = torch.tensor([0.5], requires_grad=True).float()        
            self.beta = torch.tensor([0.0], requires_grad=True).float()

        self.alpha = nn.Parameter(self.alpha )
        #self.beta = nn.Parameter(self.beta )

        self.order = 1

    def forward(self, z,dt):    

      device = "cuda" if torch.cuda.is_available() else "cpu"
      dt = torch.tensor([dt], requires_grad=False).float().to(device)

      y1 = z

      dt = dt

      y_hat= y1 - dt*self.alpha*y1

      return  y_hat


def getModel(name, init_phys = None):

    if name == "IntegratedFire":

        return IntegratedFire()
    if name == "dyn_1storder":
        return dyn_1storder(init_phys)
    if name == "Clifford_Attractor":    
        return Clifford_Attractor()
    if name == "ODE_2ObjectsSpring":
        return ODE_2ObjectsSpring(init_phys[0], init_phys[1])
    if name == "Damped_oscillation":
        return Damped_oscillation(init_phys)
    elif name == "Oscillation":
        return Oscillation()
    elif name == "Sprin_ode":
        return Sprin_ode()
    elif name == "gravity_ode":
        return gravity_ode()
    elif name == "double_pendulum":
        return double_pendulum()
    elif name == "lineal":
        return lineal()
    
    else:
        return None
    
