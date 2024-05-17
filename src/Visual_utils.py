import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from src import custom_plots as cp
from PIL import Image, ImageDraw

import torchvision

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

    
    A[:, center_row_A:] = B
    
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
    h = 200
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

        #if ((pred.max() - pred.min()) != 0):
           
            #pred = (pred - pred.min()) / (pred.max() - pred.min())
        
        if (pred.max() > 1):
            pred[pred>1] = 1

        if (pred.min() < 0):
            pred[pred<0] = 0
        

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
        cv2.putText(cv2_frame, 'Expected', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(cv2_frame, 'Predicted', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(cv2_frame, 'error', (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)

        
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

    imagesize = 100

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, ( imagesize ,imagesize ))
    i = 0


    border = np.zeros((imagesize,imagesize), np.uint8)



    for data in loader:

        print(f'Frame {i} of {num_frames}', end="\r")

        input_Data, out_Data = data

        x0= input_Data

        x0 = x0.to(device=device, dtype=torch.float)

        x2 = out_Data.to(device=device, dtype=torch.float)

        output0,output1 = model(x0)
        output = output1
        

        # Copy tensor and send to cpu and detach
        expec = x2.to('cpu').detach().numpy().copy()
        pred = output.to('cpu').detach().numpy().copy()
        input_value = x0.to('cpu').detach().numpy().copy()
        formatted_string = "{:.4f}".format(input_value[0][0])

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
        cv2.putText(cv2_frame, 'GT', (5, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 0, 0),1, cv2.LINE_AA)
        cv2.putText(cv2_frame, 'Out', (3, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(cv2_frame, formatted_string, (3, 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.55, (255, 0, 0), 1, cv2.LINE_AA)

        
        # Add frame to the video
        video.write(cv2_frame)
        i += 1

    # Release the video writer
    video.release()
    #cv2.destroyAllWindows()
    print(f'Video saved as {video_name}')

def visualize_loader(train_dataloader):
    data_iter = iter(train_dataloader)

    # Get a batch of data
    batch_data = next(data_iter)

    for data in train_dataloader:

        grid_img = torchvision.utils.make_grid(data[0], nrow=5)

        plt.figure()

        plt.imshow(grid_img.permute(1, 2, 0))

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
    
    cp.plotMultiple( X,  'sample', 'value','Latent Space', name, styleDark = False )
def CompareLatent_end_phys(model, loader, name = 'LatentSpace_end_phy.png'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    z2_encoder_list = []
    z2_phys_list = []


    for data in loader:

        input_Data, out_Data = data

        x0 = input_Data

        x0 = x0.to(device=device, dtype=torch.float)

        outputs = model(x0)
        z2_encoder, z2_phys=outputs

        print(z2_encoder.shape)
        

        
        #z2_encoder_list.append(z2_encoder.detach().cpu().numpy()[0][0])
        #z2_phys_list.append(z2_phys.detach().cpu().numpy()[0][0]) 
        z2_encoder_list=z2_encoder.detach().cpu().numpy()[0,:,0]
        z2_phys_list=z2_phys.detach().cpu().numpy()[0,:,0]
        break


    X = []
    #X.append( { 'x': range(0, len(z2_encoder_list) ), 'y': normalize(z2_encoder_list ), 'label': 'z2_encoder_list' , 'alpha':0.5  } )
    #X.append( { 'x': range(0, len(z2_phys_list) ), 'y': normalize(z2_phys_list ), 'label': 'z2_phys_list' , 'alpha':0.5  } )
    
    X.append( { 'x': range(0, len(z2_encoder_list) ), 'y': z2_encoder_list , 'label': 'z2_encoder_list' , 'alpha':0.5  } )
    X.append( { 'x': range(0, len(z2_phys_list) ), 'y': z2_phys_list , 'label': 'z2_phys_list' , 'alpha':0.5  } )
    

    cp.plotMultiple( X,  'sample', 'value','Latent Space', name, styleDark = False )

def view_masks(model, loader, iters = 10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    masks = None
    z2_phys_list = []

    iter = 0


    for data in loader:

        input_Data, out_Data = data

        x0 = input_Data

        print(x0.shape)

        test_img = x0[0,0,:,:,:]

        print("max", test_img.max())

        
        plt.figure()

        plt.imshow(test_img.permute(2, 1, 0).detach().cpu().numpy(), vmin=0, vmax=255)
        plt.show()

        x0 = x0.to(device=device, dtype=torch.float)

        print(x0.shape)

        test_img = x0[0,0,:,:,:]

        print("max", test_img.max())

        
        plt.figure()

        plt.imshow(test_img.permute(2, 1, 0).detach().cpu().numpy(), vmin=0, vmax=255)
        plt.show()

        outputs = model(x0)
        mask = model.get_masks()

        #print(mask.shape)
        #print(x0.shape)

        #img = torch.cat((x0.squeeze(2), mask), dim=1)

        #img = torch.cat((img, x0.squeeze(2)*mask), dim=1)

        mask = mask.unsqueeze(2)
        mask = mask.repeat(1,1,3,1,1)
        img = torch.cat((x0, mask[:,0:12,:,:,:]), dim=1)

        img = torch.cat((img, mask[:,12:24,:,:,:]), dim=1)

        

        img = img.squeeze(0)
        print(img.shape)

        #grid_img = torchvision.utils.make_grid(img.permute(1,0,2,3), nrow=3)

        #img = img.permute(1,0,2,3)
        grid_img = torchvision.utils.make_grid(img, nrow=12)

        print(grid_img.shape)

        #grid_img

        plt.figure()

        plt.imshow(grid_img.permute(1, 2, 0).detach().cpu().numpy())
        plt.show()



        if iter == iters:
            break

        iter += 1

    return

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

    
    cp.plotMultiple( X,  'sample', 'value','Latent Space', name, styleDark = False )

def visualize_latent_dist(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    z2_encoder_list = None
    z2_phys_list = None


    for data in loader:

        input_Data, out_Data = data

        x0 = input_Data

        x0 = x0.to(device=device, dtype=torch.float)

        outputs = model(x0)
        z2_encoder, z2_phys=outputs

        #print(z2_encoder.shape)

        z2_encoder = z2_encoder.squeeze(0)
        z2_phys = z2_phys.squeeze(0)
        

        
        #z2_encoder_list.append(z2_encoder.detach().cpu().numpy()[0][0])
        #z2_phys_list.append(z2_phys.detach().cpu().numpy()[0][0]) 
        z2_encoder_list=z2_encoder.detach().cpu().numpy() if z2_encoder_list is None else np.concatenate((z2_encoder_list, z2_encoder.detach().cpu().numpy() ))
        z2_phys_list=z2_phys.detach().cpu().numpy() if z2_phys_list is None else np.concatenate((z2_phys_list, z2_phys.detach().cpu().numpy() ))
        


    X = []
    #X.append( { 'x': range(0, len(z2_encoder_list) ), 'y': normalize(z2_encoder_list ), 'label': 'z2_encoder_list' , 'alpha':0.5  } )
    #X.append( { 'x': range(0, len(z2_phys_list) ), 'y': normalize(z2_phys_list ), 'label': 'z2_phys_list' , 'alpha':0.5  }
    #print(z2_encoder_list.shape)
    n_epoch = z2_encoder_list.shape[0]
    d_dim = z2_encoder_list.shape[1]

    print("n_epoch",n_epoch)
    print("d_dim",d_dim)

    for i in range(d_dim):
    
        X.append( { 'x': range(0, n_epoch ), 'y': z2_encoder_list[:,i] , 'label': 'z2_encoder_list_'+str(i) , 'alpha':0.5  } )
        #X.append( { 'x': range(0, n_epoch ), 'y': z2_phys_list[:,i] , 'label': 'z2_phys_list_'+str(i) , 'alpha':0.5  } )
        
    cp.plotMultiple( X,  'sample', 'value','Latent Space', "data distribution", show=True, styleDark = False, plot_type='hist'	 )
    X = []
    
    for i in range(d_dim):
        X.append( { 'x': range(0, 500 ), 'y': z2_encoder_list[::10,i] , 'label': 'z2_encoder_list_'+str(i) , 'alpha':0.5  } )
        print("max",i,z2_encoder_list[:,i].max())
        print("min",i,z2_encoder_list[:,i].min())
        
    cp.plotMultiple( X,  'sample', 'value','Latent Space', "data distribution", show=True,styleDark = False, plot_type='scatter' )
    cp.plotMultiple( X,  'sample', 'value','Latent Space', "data distribution", show=True,styleDark = False, plot_type='plot' )

    

def get_center_mass(image):
    
    image = image.detach().cpu().numpy()*255
    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calculate moments
    moments = cv2.moments(binary_mask)

    # Calculate center of mass
    if moments['m00'] != 0:
        center_of_mass_x = int(moments['m10'] / moments['m00'])
        center_of_mass_y = int(moments['m01'] / moments['m00'])
        center_of_mass = [center_of_mass_x, center_of_mass_y]
        #print("Center of Mass:", center_of_mass)
    else:
        print("No center of mass found, image is likely empty.")

    return np.array(center_of_mass)

def vis_frame_masks(img):

    img_hor = None

    for  f in range(img.shape[0]):

        m1 = img[f,0,:,:].detach().cpu().numpy() 
        m2 = img[f,1,:,:].detach().cpu().numpy() 
        

        img_ver = np.concatenate((m1, m2), axis=0)

        img_ver = np.concatenate((img_ver, np.abs(m1+m2)), axis=0)

        img_hor = img_ver if f == 0 else np.concatenate((img_hor, img_ver), axis=1)

    plt.figure()
    plt.imshow(img_hor, cmap='gray')
    plt.show()
        
def visualize_cm(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    z2_encoder_list = None
    z2_phys_list = None


    iter = 0

    for data in loader:

        iter += 1

        if iter != 9:
                continue

        input_Data, out_Data = data

        x0 = input_Data

        x0 = x0.to(device=device, dtype=torch.float)

        vis_frame_masks(x0[0])

        list_cm = None

        for frame in range(0,x0.shape[1]):

            

            img_0 = x0[0,frame,0,:,:]
            img_1 = x0[0,frame,1,:,:]
            
            cm_0 = get_center_mass(img_0)
            cm_1 = get_center_mass(img_1)

            

            cm = np.concatenate((cm_0, cm_1))

          

            list_cm = cm if frame == 0 else np.vstack((list_cm, cm))


        outputs = model(x0)
        z2_encoder, z2_phys=outputs

        #print(z2_encoder.shape)

        z2_encoder = z2_encoder.squeeze(0)
        z2_phys = z2_phys.squeeze(0)
        

        
        #z2_encoder_list.append(z2_encoder.detach().cpu().numpy()[0][0])
        #z2_phys_list.append(z2_phys.detach().cpu().numpy()[0][0]) 
        z2_encoder_list=z2_encoder.detach().cpu().numpy() 
        z2_phys_list=z2_phys.detach().cpu().numpy() 

        break

  
    z2_encoder_list  = z2_encoder_list 

    print(z2_encoder_list)
    print(list_cm)

    y1 = z2_encoder_list.max()
    y0 = z2_encoder_list.min()

    x1 = list_cm.max()  
    x0 = list_cm.min()

    m = (y1-y0)/(x1-x0)

    b = y1 - m*x1

    print("m",m)
    print("m_12",m*12)

    #z2_encoder_list = (1/m) * (z2_encoder_list - b)
    #z2_phys_list = (1/m) * (z2_phys_list - b)

    plt.figure()

    t = range(z2_encoder_list.shape[0])
    plt.plot(t,z2_encoder_list[:,0], '-b', label='z2_encoder_list_0')
    plt.plot(t,z2_encoder_list[:,1], '-g',label='z2_encoder_list_1')
    plt.plot(t,z2_encoder_list[:,2], '-y',label='z2_encoder_list_2')
    plt.plot(t,z2_encoder_list[:,3], '-r',label='z2_encoder_list_3')
    plt.legend()
    plt.show()

    plt.plot(t,z2_phys_list[:,0], '-b', label='z2_phy_list_0')
    plt.plot(t,z2_phys_list[:,1], '-g',label='z2_phy_list_1')
    plt.plot(t,z2_phys_list[:,2], '-y',label='z2_phy_list_2')
    plt.plot(t,z2_phys_list[:,3], '-r',label='z2_phy_list_3')
    plt.legend()
    plt.show()

    plt.plot(t,list_cm[:,0], '--b',label='center_mass_0')
    plt.plot(t,list_cm[:,1], '--g',label='center_mass_1')
    plt.plot(t,list_cm[:,2], '--y',label='center_mass_2')
    plt.plot(t,list_cm[:,3], '--r',label='center_mass_3')

    plt.legend()
    plt.show()

    plt.plot(z2_encoder_list[:,0],z2_encoder_list[:,1], '--b',label='center_mass_0')
    plt.plot(z2_encoder_list[:,2],z2_encoder_list[:,3], '--r',label='center_mass_1')
    
    plt.legend()
    plt.show()

    plt.plot(list_cm[:,0],list_cm[:,1], '--b',label='center_mass_0')
    plt.plot(list_cm[:,2],list_cm[:,3], '--r',label='center_mass_1')
    
    plt.legend()
    plt.show()
def normalize(x):
    # normalize list of values to max value 1 and mean 0

    # list to numpy array
    x = np.array(x)
    try :
        # normalize
        return (x - x.min()) / (x.max() - x.min())
    except:

        return np.array(x)


def generateVideo(a,DynamicsType, name):
    

    # Create a video of the swinging pendulum
    video_name = name+'.mp4'
    frame_rate = 30
    duration = len(a) / frame_rate
    num_frames = frame_rate * duration

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, (50, 50))

    if DynamicsType == "Scale":
        ImageGenerator = create_scale_Image
    if DynamicsType == "Intensity":
        ImageGenerator = create_intensity_image
    if DynamicsType == "Motion":
        ImageGenerator = create_pendulum_image

    


    # Generate frames and add to video
    for angle in a:
        # Create pendulum image
        frame_image = ImageGenerator(angle)

        #clipped_img_array = np.clip(frame_image, 0, 255)

        # If you want to convert the data type to uint8 after clipping
        #clipped_img_array = clipped_img_array.astype(np.uint8)

        #if(len(frame_image.shape) < 3):
         #   frame_image = np.stack((frame_image,)*3, axis=-1)
        
        clipped_img_array = frame_image * 255
        clipped_img_array = clipped_img_array.astype(np.uint8)
        
        # Convert PIL image to OpenCV format
        cv2_frame = cv2.cvtColor(clipped_img_array, cv2.COLOR_RGB2BGR)
        
        # Add frame to the video
        video.write(cv2_frame)

    # Release the video writer
    video.release()