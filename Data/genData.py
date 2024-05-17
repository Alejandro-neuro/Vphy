import numpy as np
import math
import matplotlib.pyplot as plt
from src import custom_plots as cp
import pandas as pd
from src import custom_plots as cp
from PIL import Image, ImageDraw
import math


from omegaconf import OmegaConf

cfg = OmegaConf.load("config.yaml")
Image_size = cfg.image_size

def generatePendulumA(g,L,a0, a1):

    t = np.arange(0,30,1/30)
    a = (np.cos(5*t)*np.exp(-.1*t) )*np.pi/2
    
    #X = []
    #X.append( { 'x': t, 'y': a, 'label': 'Pendulum angle'} )
    #cp.plotMultiple( X,  'time (ms)', 'Angle','Pendulum angle', 'test', styleDark = False )
    
    return t,a    

def generateDynamics(max=1, min=0, dt = 1/100):
    t = np.arange(0,54, dt)
    m = (max-min)/(1-(-1))
    b= max - m
    #a = m*np.exp(-0.02*t)*np.cos(2*t)+b
    a = m*np.exp(-0.04*t)*np.cos(2*t)+b
    #a = m*np.cos(2*t)+b
    
    #X = []
    #X.append( { 'x': t, 'y': a, 'label': 'Intensity'} )
    #cp.plotMultiple( X,  'time (ms)', 'Intensity','Intensity', 'test', styleDark = False )
    
    return t,a  
class FitzHugh_Nagumo:
    def __init__(self, a=0.7, b=0.8, tau=12.5, I=0.5, dt=0.01):
        self.a = a
        self.b = b
        self.tau = tau
        self.I = I
        self.dt = dt

    def step(self, v, w):
        v_new = v + self.dt * (self.a * v - self.b * v ** 3 - w + self.I)
        w_new = w + self.dt * (v - self.tau * w)
        return v_new, w_new

    def simulate(self, v0, w0, steps):
        v = v0
        w = w0
        v_list = [v]
        w_list = [w]
        for _ in range(steps):
            v, w = self.step(v, w)
            v_list.append(v)
            w_list.append(w)
        return v_list, w_list
    
def generateFitzHughNagumo(a,DynamicsType, a0, a1):
    fitzHugh_Nagumo_test = FitzHugh_Nagumo()
    v0 = 0.1
    w0 = 0.1
    steps = 1000
    v, w = fitzHugh_Nagumo_test.simulate(v0, w0, steps)

    #X = []
    #X.append( { 'x': np.arange(0,steps+1), 'y': v, 'label': 'v'} )
    #X.append( { 'x': np.arange(0,steps+1), 'y': w, 'label': 'w'} )
    #cp.plotMultiple( X,  'time (ms)', 'Intensity','FitzHugh Nagumo', 'test', styleDark = True )

    return v
    
def generateIntegrated_Fire_model(max=1, min=0.2):
    # Parameters for the integrate-and-fire model
    tau_m = 10  # Membrane time constant (ms)
    V_rest = -65  # Resting membrane potential (mV)
    V_th = -50  # Threshold potential (mV)
    V_reset = -70  # Reset potential after spike (mV)
    R = 1  # Membrane resistance (Ohm)
    I = 2  # Applied current (nA)

    # Simulation parameters
    dt = 0.01  # Time step (ms)
    t_max = 100  # Total simulation time (ms)
    num_steps = int(t_max / dt)  # Number of simulation steps

    # Initialize variables for integrate-and-fire model
    v_IF = V_rest  # Initial membrane potential (mV)

    # Arrays to store simulation results
    time = np.arange(0, t_max, dt)
    I = np.sin(2 * np.pi * time / 50)  # Sinusoidal input current
    I[I < 0] = 0  # Half-wave rectification
    I[I > 0] = 1  # Double the current for positive values
    v_integrate_fire = np.zeros_like(time)

    for i in range(num_steps):
   

        # Integrate-and-fire model dynamics
        dv_IF = (V_rest - v_IF + R * I[i]) / tau_m * dt
        v_IF += dv_IF
        if v_IF >= V_th:
            v_IF = V_reset

        # Store membrane potentials
        v_integrate_fire[i] = v_IF

    v_max = np.max(v_integrate_fire)
    v_min = np.min(v_integrate_fire)
    m = (max-min)/(v_max-v_min)
    b = max - m*v_max

    v_norm = m*v_integrate_fire + b

    return time, v_norm

def create_intensity_image(I, noise = False, shapeType = 'simple', base = None):
    if base is None:
        if shapeType == 'complex':
            base = Image.open('Data/neuron2.png').convert('L')
        elif shapeType == 'simple':
            base = Image.open('Data/neuron_gr.png').convert('L')

    # Create a blank gray image

    width, height = Image_size, Image_size
    img_array = np.array(base)

    #resize image
    img_array = Image.fromarray(img_array)
    img_array = img_array.resize((width, height))
    img_array = np.array(img_array)
    #img_array = cv2.resize(img_array, (width, height))

    img_array = img_array / np.max(img_array)


    if shapeType == 'simple':
        img_array[img_array < 0.5] = 0
        img_array[img_array >= 0.5] = I

    if shapeType == 'complex':        
        img_array = img_array* I
    if False:
        bknoise = [[1,1], [1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]]
        for bk in bknoise:
            img_array[bk] = 1
    # generate random gaussian noise matrix proportional to the intensity
    if noise:
        img_array = img_array + np.random.normal(0, 0.1, img_array.shape) * I

    img_array[img_array >= 1] = 1
    

    return img_array

def create_scale_image(I, noise = False, shapeType = 'complex', base = None):

    """
    Generate an image with a circle centered in it.

    :param image_size: Size of the image (tuple of width and height)
    :param radius: Radius of the circle
    :return: Numpy array representing the image with the circle
    """

    image_size = (Image_size, Image_size)

    # Create a blank image
    image = np.zeros(image_size)

    radius = 5 + I*10

    # Get center coordinates
    center_x = image_size[0] // 2
    center_y = image_size[1] // 2

    # Generate grid of coordinates
    X, Y = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))

    # Compute distance from center for each pixel
    distances = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    # Set pixels inside the circle to 1
    image[distances <= radius] = 1

    return image

def create_half_radius_circle_image(X, n=50):
    # Create an empty image
    image = np.zeros((n, n), dtype=np.float32)
    
    # Center of the image
    center = (n // 2, n // 2)
    
    # Radii for the left and right halves
    radius_left = 10 + X
    radius_right = 10 - X
    
    # Draw the circle
    for i in range(n):
        for j in range(n):
            distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            if j < center[1]:  # Left half
                if distance <= radius_left:
                    image[i, j] = 1.0
            else:  # Right half
                if distance <= radius_right:
                    image[i, j] = 1.0
    
    return image

def create_pendulum_image(I, noise = False, shapeType = 'complex', base = None):
    # Create a blank image
    width, height = Image_size, Image_size
    image = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(image)

    # Pendulum parameters
    pendulum_length = 20
    bob_radius = 3

    # Calculate bob position
    #angle_rad = math.radians(angle_deg)
    angle_rad = math.radians(I*180)
    bob_x = width // 2 + pendulum_length * math.sin(angle_rad)
    bob_y = height // 2 + pendulum_length * math.cos(angle_rad)

    # Draw pendulum rod
    draw.line([(width // 2, height // 2), (bob_x, bob_y)], fill='grey', width=2)

    # Draw pendulum bob
    draw.ellipse((bob_x - bob_radius, bob_y - bob_radius, bob_x + bob_radius, bob_y + bob_radius), fill='white')


    # Setting the points for cropped image
    left = 0
    top = (height // 2)-10
    right = width
    bottom = height
    
    # Cropped image of above dimension
    # (It will not change original image)
    #image = image.crop((left, top, right, bottom))

    # transform image black & white
    image = image.convert('L')

    # Save the image
    #image.save(f'pendulum_{angle_deg}deg.png')

    return np.array(image)/255