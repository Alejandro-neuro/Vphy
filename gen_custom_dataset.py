import numpy as np
import matplotlib.pyplot as plt

import cv2


p0 = np.array([22,22])
p0_past = np.array([20,20])

p1 = np.array([32, 32])
p1_past = np.array([34, 34])

def force(p0, p1):
    k = 2
    L = 12
    F = k*(np.linalg.norm(p0-p1) - L)*(p1-p0)/np.linalg.norm(p0-p1)
    return F

def prediction(p0, p1, p0_past, p1_past, F):
    dt = 0.1
    p0_new = 2*p0 - p0_past + F*dt*dt
    p1_new = 2*p1 - p1_past - F*dt*dt
    return p0_new, p1_new

p0_list = []
p1_list = []

p0_list.append(p0_past)
p1_list.append(p1_past)

p0_list.append(p0)
p1_list.append(p1)

for i in range(1000):
    F = force(p0, p1)
    p0_new, p1_new = prediction(p0, p1, p0_past, p1_past, F)
    p0_past = p0
    p1_past = p1
    p0 = p0_new
    p1 = p1_new
    p0_list.append(p0)
    p1_list.append(p1)

p0_np = np.array(p0_list)
p1_np = np.array(p1_list)

print(p0_np[:,1].max())
print(p1_np[:,1].max())

print(p0_np[:,1].min())
print(p1_np[:,1].min())

t = np.arange(p0_np[:,0].shape[0])
print(t.shape)

ts = t[0::3]
print(ts.shape)
p_s = p0_np[0::3,0]
print(p_s.shape)

plt.plot(p0_np[:,0],'r', label = '0_x' )
plt.scatter(ts,p_s )
plt.plot(p0_np[:,1],'b', label = '0_y' )
plt.plot(p1_np[:,0],'g', label = '1_x' )
plt.plot(p1_np[:,1],'y', label = '1_y' )
plt.legend()
plt.show()


#from keras.datasets import mnist
#data = mnist.load_data()

#import draw


def create_mnist_spring_frame(p1,p2, number0=[], number5 =[]):
    #Create a blank image
    mask1 = np.zeros((64,64))
    mask2 = np.zeros((64,64))

    size_obj = 22
    half= size_obj//2


    #create square 1
    # if number0.any() == None:
    #   s1 = np.ones((size_obj,size_obj))
    # else:
    s1 = number0

    #create square 2
    # if number5.any() == None:
    #   s2 = np.ones((size_obj,size_obj))
    # else:
    s2 = number5

    #aling squares to img
    # p1 = p1.astype(int)
    # p2 = p2.astype(int)
    # s1 = s1.astype(int)
    # s2 = s2.astype(int)

    #print(p1)
    #print(p2)

    #print([p1[0]-half,p1[0]+half, p1[1]-half,p1[1]+half ])

    mask1[int(p1[0]-half):int(p1[0]+half), int(p1[1]-half):int(p1[1]+half) ] = s1
    mask2[int(p2[0]-half):int(p2[0]+half), int(p2[1]-half):int(p2[1]+half) ] = s2

    mask1  = mask1.reshape(1,64,64)
    mask2  = mask2.reshape(1,64,64)

    sum_mask = mask1 + mask2


    img = np.concatenate((mask1,mask2,sum_mask), axis=0)



    return img

def create_mnist_spring_video(p1_list,p2_list, number0=[], number5 =[]):

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    h = 64
    frame_rate = 2


    frames = []
    for i in range(len(p1_list)):

        #print(i)
        ##print(i)
        p1 = p1_list[i]
        p2 = p2_list[i]

        frame = create_mnist_spring_frame(p1,p2,number0, number5)
        frame = frame.transpose(1,2,0)
        frames.append(frame)

    # create video
    video = cv2.VideoWriter('./Custom_mnist.mp4', fourcc, frame_rate, ( h ,h ))
    for i in range(len(frames)):

        #print(i)



        frame = frames[i]
        #print(frame.shape)
        #print(frame.max())
        #break
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = np.uint8(frame*255)
        frame = frame*255
        frame[frame>255] = 128
        frame = frame.astype('uint8')

        video.write(frame)
    video.release()



    return frames

import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
digits = x_train[0:2, 3:-3, 3:-3]/255

p1 = np.array([20,20])
p2 = np.array([34,34])
frame = create_mnist_spring_frame(p1,p2, digits[0] , digits[1])

frame = frame.transpose(1,2,0)
plt.figure()
plt.imshow(frame)
plt.show()

frames = create_mnist_spring_video(p0_np,p1_np, digits[0] , digits[1])



np_frames = np.array(frames)
dataset_custom = []
for i in range(np_frames.shape[0]-(3*10)):

  dataset_custom.append( np_frames[i:i+(3*10):3] )

np_dataset = np.array(dataset_custom)
print(np_dataset.shape)


np.save("custom_dataset", np_dataset)