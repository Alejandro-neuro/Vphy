import numpy as np
import math
import matplotlib.pyplot as plt
from src import custom_plots as cp
from src import Visual_utils as vu
import pandas as pd
from src import custom_plots as cp
from PIL import Image, ImageDraw
import math
import cv2


def generatePendulumA(g,L,a0, a1):

    t = np.arange(0,30,1/30)
    a = (np.cos(5*t)*np.exp(-.1*t) )*np.pi/2
    
    X = []
    X.append( { 'x': t, 'y': a, 'label': 'Pendulum angle'} )
    cp.plotMultiple( X,  'time (ms)', 'Angle','Pendulum angle', 'test', styleDark = True )
    
    return t,a    

def generateIntensity():

    t = np.arange(0,10,1/100)
    t = np.arange(0,10, 0.1)
    m = (1-0.2)/(1-(-1))
    b= 1 - m
    a = m*np.cos(t)+b
    
    X = []
    X.append( { 'x': t, 'y': a, 'label': 'Intensity'} )
    cp.plotMultiple( X,  'time (ms)', 'Intensity','Intensity', 'test', styleDark = True )
    
    return t,a  
def create_intensity_image(I, noise = False, shapeType = 'complex', base = None):
    if base is None:
        if shapeType == 'complex':
            base = Image.open('Data/neuron2.png').convert('L')
        elif shapeType == 'simple':
            base = Image.open('Data/neuron_gr.png').convert('L')

    # Create a blank gray image

    width, height = 50, 50
    img_array = np.array(base)

    img_array = img_array / np.max(img_array)


    if shapeType == 'simple':
        img_array[img_array < 0.5] = 0
        img_array[img_array >= 0.5] = I

    if shapeType == 'complex':        
        img_array = img_array* I

    bknoise = [[1,1], [1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]]
    for bk in bknoise:
        img_array[bk] = 1
    # generate random gaussian noise matrix proportional to the intensity
    if noise:
        img_array = img_array + np.random.normal(0, 0.1, img_array.shape) * I

    img_array[img_array >= 1] = 1
    

    return img_array

def create_pendulum_image(angle_deg):
    # Create a blank image
    width, height = 500, 500
    image = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(image)

    # Pendulum parameters
    pendulum_length = 200
    bob_radius = 10

    # Calculate bob position
    #angle_rad = math.radians(angle_deg)
    angle_rad = angle_deg
    bob_x = width // 2 + pendulum_length * math.sin(angle_rad)
    bob_y = height // 2 + pendulum_length * math.cos(angle_rad)

    # Draw pendulum rod
    draw.line([(width // 2, height // 2), (bob_x, bob_y)], fill='grey', width=3)

    # Draw pendulum bob
    draw.ellipse((bob_x - bob_radius, bob_y - bob_radius, bob_x + bob_radius, bob_y + bob_radius), fill='white')


    # Setting the points for cropped image
    left = 0
    top = (height // 2)-10
    right = width
    bottom = height
    
    # Cropped image of above dimension
    # (It will not change original image)
    image = image.crop((left, top, right, bottom))

    # transform image black & white
    image = image.convert('L')

    # Save the image
    #image.save(f'pendulum_{angle_deg}deg.png')

    return image
    #image.show()

def generateVideo():
    g=9.81
    L=1
    x0=np.pi/2
    x1=np.pi/2.5
    t,a = generatePendulumA(g,L,x0, x1)

    # Create a video of the swinging pendulum
    video_name = 'pendulum_swing2.mp4'
    frame_rate = 30
    duration = len(a) / frame_rate
    num_frames = frame_rate * duration

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, (500, 260))

    


    # Generate frames and add to video
    for angle in a:
        # Create pendulum image
        frame_image = create_pendulum_image(angle)
        
        # Convert PIL image to OpenCV format
        cv2_frame = cv2.cvtColor(np.array(frame_image), cv2.COLOR_RGB2BGR)
        
        # Add frame to the video
        video.write(cv2_frame)

    # Release the video writer
    video.release()

