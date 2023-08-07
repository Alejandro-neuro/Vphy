import numpy as np
import cv2
import matplotlib.pyplot as plt

def createImage(pos,x ):
    nArray = np.zeros((20, 20,3), np.uint8) #Create the arbitrary input img


    #create an empty list of colors with the same length as the number of rows in x



    color = np.zeros((x.shape[0],3), dtype=int)

    color[x == 0 , :] = [50,50,50]
    color[np.where((x > 0) & (x < 1)) , :] = [128,128,128]
    color[x == 1 , :] = [255,255,255]

   

    for i in range(len(pos)):
        p = ( int(pos[i][0]), int(pos[i][1]) )
        nArray = cv2.circle(nArray, p, 2, color = (255, 133, 233), thickness=-1)
    
    plt.imshow(nArray)
