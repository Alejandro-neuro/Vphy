import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from src import custom_plots as cp
from PIL import Image, ImageDraw

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
def set_center_values(A, B):
    center_row_A = A.shape[0] // 2
    center_col_A = A.shape[1] // 2
    center_row_B = B.shape[0] // 2
    center_col_B = B.shape[1] // 2
    
    start_row_A = center_row_A - center_row_B
    start_col_A = center_col_A - center_col_B
    end_row_A = start_row_A + B.shape[0]
    end_col_A = start_col_A + B.shape[1]
    
    A[start_row_A:end_row_A, start_col_A:end_col_A] = B
    
    return A



def visualize(model, loader, video_name = 'ExpVsPred.mp4'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Create a video of the swinging pendulum
    frame_rate = 30
    duration = len(loader) / frame_rate
    num_frames = frame_rate * duration

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h = 300
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, ( h ,h ))
    i = 0


    border = np.zeros((h,h), np.uint8)



    for data in loader:

        print(f'Frame {i} of {num_frames}', end="\r")

        input_Data, out_Data = data

        x0= input_Data

        x0 = x0.to(device=device, dtype=torch.float)

        x2 = out_Data.to(device=device, dtype=torch.float)

        outputs = model(x0)
        z,rec=outputs
        rec0,rec1,outrec=rec

        # Copy tensor and send to cpu and detach
        expec = x2.to('cpu').detach().numpy().copy()
        pred = outrec.to('cpu').detach().numpy().copy()

        # squeeze to remove batch dimension
        expec = np.squeeze(expec)
        pred = np.squeeze(pred)

        #print(expec.shape)
        #print(pred.shape)

        # permute to (height, width, channels)
        #expec = np.transpose(expec, (1, 2, 0))
        #pred = np.transpose(pred, (1, 2, 0))

        #print(expec.shape)
        #print(pred.shape)

        # normalize to 0-1

        if ((pred.max() - pred.min()) != 0):
           
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        

        # concatenate expected and predicted one over the other
        expected_pred = np.concatenate((expec, pred), axis=0)

        expected_pred = np.concatenate((expected_pred, np.abs(expec-pred) ), axis=0)

      

        #matrix to cv2 image
        expected_pred = np.uint8( np.round(expected_pred * 255, 0) )   

        #replicate the image 3 times to have 3 channels 
        expected_pred = set_center_values(border, expected_pred)
        expected_pred = np.repeat(expected_pred[:, :, np.newaxis], 3, axis=2)

        # Convert PIL image to OpenCV format
        cv2_frame = cv2.cvtColor(expected_pred, cv2.COLOR_RGB2BGR)       
        
        # add text label
        cv2.putText(cv2_frame, 'Expected', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(cv2_frame, 'Predicted', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(cv2_frame, 'error', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 1, cv2.LINE_AA)

        
        # Add frame to the video
        video.write(cv2_frame)
        i += 1

    # Release the video writer
    video.release()
    #cv2.destroyAllWindows()
    print(f'Video saved as {video_name}')


def visualize_dec(model, loader, video_name = 'ExpVsPred.mp4'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Create a video of the swinging pendulum
    frame_rate = 30
    duration = len(loader) / frame_rate
    num_frames = frame_rate * duration

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, ( 600 ,600 ))
    i = 0


    border = np.zeros((600,600), np.uint8)



    for data in loader:

        print(f'Frame {i} of {num_frames}', end="\r")

        input_Data, out_Data = data

        x0= input_Data

        x0 = x0.to(device=device, dtype=torch.float)

        x2 = out_Data.to(device=device, dtype=torch.float)

        output = model(x0)
        

        # Copy tensor and send to cpu and detach
        expec = x2.to('cpu').detach().numpy().copy()
        pred = output.to('cpu').detach().numpy().copy()

        # squeeze to remove batch dimension
        expec = np.squeeze(expec)
        pred = np.squeeze(pred)

        if ((pred.max() - pred.min()) != 0):
           
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        

        expected_pred = np.concatenate((expec, pred), axis=0)

      

        #matrix to cv2 image
        expected_pred = np.uint8( np.round(expected_pred * 255, 0) )   

        #replicate the image 3 times to have 3 channels 
        expected_pred = set_center_values(border, expected_pred)
        expected_pred = np.repeat(expected_pred[:, :, np.newaxis], 3, axis=2)

        # Convert PIL image to OpenCV format
        cv2_frame = cv2.cvtColor(expected_pred, cv2.COLOR_RGB2BGR)       
        
        # add text label
        cv2.putText(cv2_frame, 'Expected', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(cv2_frame, 'Predicted', (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        
        # Add frame to the video
        video.write(cv2_frame)
        i += 1

    # Release the video writer
    video.release()
    #cv2.destroyAllWindows()
    print(f'Video saved as {video_name}')



def CompareLatent(model, loader, name = 'LatentSpace.png'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    z0_list = []
    z1_list = []
    z2_list = []


    for data in loader:

        input_Data, out_Data = data

        x0 = input_Data

        x0 = x0.to(device=device, dtype=torch.float)

        x2 = out_Data.to(device=device, dtype=torch.float)

        outputs = model(x0)
        z,rec=outputs
        rec0,rec1,outrec=rec

        z0,z1,z2 = z 
        z0_list.append(z0.detach().cpu().numpy()[0][0])
        z1_list.append(z1.detach().cpu().numpy()[0][0]) 
        z2_list.append(z2.detach().cpu().numpy()[0][0])


    X = []
    X.append( { 'x': range(0, len(z0_list) ), 'y': normalize(z0_list ), 'label': 'z0' , 'alpha':0.5  } )
    X.append( { 'x': range(0, len(z1_list) ), 'y': normalize(z1_list ), 'label': 'z1' , 'alpha':0.5  } )
    X.append( { 'x': range(0, len(z2_list) ), 'y': normalize(z2_list ), 'label': 'z2' , 'alpha':0.5  } )
    
    cp.plotMultiple( X,  'sample', 'value','Latent Space', name, styleDark = True )

def CompareError(model, loader, name = 'ErrorImg.png'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    z0_list = []
    z1_list = []
    z2_list = []


    for data in loader:

        input_Data, out_Data = data

        input_img = input_Data

        input_img = x0.to(device=device, dtype=torch.float)

        x0=input_img[:,-2,:,:]
        x1=input_img[:,-1,:,:]

        x2 = out_Data.to(device=device, dtype=torch.float)

        outputs = model(x0)
        z,rec=outputs
        rec0,rec1,outrec=rec

        lossMSE = nn.MSELoss()
        
        l_in=lossMSE(outrec, x1)
        l_out=lossMSE(outrec,x2)

        
        z0_list.append(l_in.detach().cpu().numpy())
        z1_list.append(l_out.detach().cpu().numpy()) 


    X = []
    X.append( { 'x': range(0, len(z0_list) ), 'y': normalize(z0_list ), 'label': 'Error Input' , 'alpha':0.5  } )
    X.append( { 'x': range(0, len(z1_list) ), 'y': normalize(z1_list ), 'label': 'Error recontruction' , 'alpha':0.5  } )

    
    cp.plotMultiple( X,  'sample', 'value','Latent Space', name, styleDark = True )

def normalize(x):
    # normalize list of values to max value 1 and mean 0

    # list to numpy array
    x = np.array(x)
    try :
        # normalize
        return (x - x.min()) / (x.max() - x.min())
    except:

        return np.array(x)
