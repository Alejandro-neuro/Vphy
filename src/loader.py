import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score

from omegaconf import OmegaConf
import Data.genData as genData
from torchvision import transforms

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw



  
class Dataset(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, x, dt, type, nInFrames = 3 ,sr = 10 , noise=True, shapeType='complex', transform=None):
            'Initialization'
            self.x = x
            self.dt = dt
            self.transform = None
            self.convert_tensor = transforms.ToTensor()
            self.nInFrames = nInFrames
            self.base = None
            self.sr = sr
            self.type = type

            self.noise = noise
            self.shapeType = shapeType

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.x)-self.nInFrames*self.sr 

      def __getitem__(self, index):
            'Generates one sample of data'

            if self.type == "Motion":
                  ImageGenerator = genData.create_pendulum_image
            if self.type == "Scale":
                  ImageGenerator = genData.create_scale_image
            if self.type == "Intensity":
                  ImageGenerator = genData.create_intensity_image
            # Concatenate 3 frames to create 3D image
            for i in range(self.nInFrames):
                  x_temp = self.convert_tensor(ImageGenerator( self.x[index+i*self.sr], noise=self.noise, shapeType=self.shapeType  ))
                  
                  x_temp = x_temp.unsqueeze(0)
                 
                  if i == 0:  
                        input = x_temp
                  else :
                        input = torch.cat((input, x_temp), 0)
            # Select sample                


            out =self.convert_tensor( ImageGenerator( self.x[index + self.nInFrames*self.sr ], noise=self.noise, shapeType=self.shapeType   ) )


            if self.transform:
                  input = self.transform(input)
                  out = self.transform(out)


            return input, out

class Dataset_decoder(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, x, dynamics_type, nInFrames = 3 ,sr = 10 , noise=True, shapeType='complex', transform=None):
            'Initialization'
            self.x = x
            self.transform = None
            self.convert_tensor = transforms.ToTensor()
            self.nInFrames = nInFrames
            self.base = None
            self.sr = sr
            self.type = type

            self.dynamics_type=dynamics_type

            self.noise = noise
            self.shapeType = shapeType

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.x)

      def __getitem__(self, index):
            'Generates one sample of data'

            if self.dynamics_type == "Motion":
                  ImageGenerator = genData.create_pendulum_image
            if self.dynamics_type == "Scale":
                  ImageGenerator = genData.create_scale_image
            if self.dynamics_type == "Intensity":
                  ImageGenerator = genData.create_intensity_image
            
            input = torch.tensor([self.x[index]])   
            out =self.convert_tensor(ImageGenerator( self.x[index], noise=self.noise, shapeType=self.shapeType  ))


            if self.transform:
                  input = self.transform(input)
                  out = self.transform(out)


            return input, out
      
class Dataset_from_folder(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, x):
            'Initialization'
            self.x = x
            self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1) ])
            self.convert_tensor = transforms.ToTensor()

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.x)

      def __getitem__(self, index):
            'Generates one sample of data'     
            
            input = torch.from_numpy(self.x[index].transpose( (0,3,1,2)))

            if self.transform:
                  input = self.transform(input)
                  out = self.transform(input)

            return input, out
            

def getLoader(X, type , split = True,   dt=1/100, nInFrames = 3,sr = 10 ,  noise=True, shapeType='simple'):   

      if split:     
            #split dataset 80-20 for training and validation

            train_x, val_x = train_test_split(X, test_size=0.2, shuffle=False)

            #create train and test dataloaders

            train_dataset = DataLoader( Dataset(train_x, dt=dt, type=type, nInFrames = nInFrames,sr = sr ,  noise=noise, shapeType=shapeType), batch_size=32, shuffle=True)
            val_dataset = DataLoader( Dataset(val_x, dt=dt, type=type, nInFrames = nInFrames,sr = sr ,  noise=noise, shapeType=shapeType), batch_size=32, shuffle=True)    

            return train_dataset, val_dataset, train_x, val_x 
      else :
            return DataLoader( Dataset(X, dt=dt, type=type, nInFrames = nInFrames,sr = 10 ,  noise=noise, shapeType=shapeType), batch_size=1, shuffle=False)

def getLoader_decoder(X, type , split = True,   dt=1/100, nInFrames = 3,sr = 10 ,  noise=False, shapeType='simple', batch_size=14):   

      if split:     
            #split dataset 80-20 for training and validation

            train_x, val_x = train_test_split(X, test_size=0.2, shuffle=False)

            #create train and test dataloaders

            train_dataset = DataLoader( Dataset_decoder(train_x, type, nInFrames = nInFrames,sr = sr ,  noise=noise, shapeType=shapeType), batch_size=batch_size, shuffle=False)
            val_dataset = DataLoader( Dataset_decoder(val_x,  type, nInFrames = nInFrames,sr = sr ,  noise=noise, shapeType=shapeType), batch_size=batch_size, shuffle=False)    

            return train_dataset, val_dataset, train_x, val_x 
      else :
            return DataLoader( Dataset_decoder(X, type, nInFrames = nInFrames,sr = 10 ,  noise=noise, shapeType=shapeType), batch_size=1, shuffle=False)     


def getLoader_folder(X, split = True):   

      if split:     
            #split dataset 80-20 for training and validation

            train_x, val_x = train_test_split(X, test_size=0.2, shuffle=False)

            #create train and test dataloaders

            train_dataset = DataLoader( Dataset_from_folder(train_x), batch_size=32, shuffle=True)
            val_dataset = DataLoader( Dataset_from_folder(val_x), batch_size=32, shuffle=False)    

            return train_dataset, val_dataset, train_x, val_x 
      else :
            return DataLoader( Dataset_from_folder(X), batch_size=1, shuffle=False)
      

def get_template():

      convert_tensor = transforms.ToTensor()
      img = genData.create_pendulum_image( 0 )
      # show image
      plt.imshow(img)
      return convert_tensor( genData.create_pendulum_image( 0 ) )
      

if __name__ == "__main__":
    pass
