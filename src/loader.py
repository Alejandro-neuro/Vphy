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


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x, dt,  transform=None):
        'Initialization'
        self.x = x
        self.dt = dt
        self.transform = None
        self.convert_tensor = transforms.ToTensor()

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)-3

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x0 =self.convert_tensor( genData.create_pendulum_image( self.x[index] ) )
        
        x1 =self.convert_tensor( genData.create_pendulum_image( self.x[index+1] ) )
  
        x2 =self.convert_tensor( genData.create_pendulum_image( self.x[index+2] ) )


        if self.transform:
            x0 = self.transform(x0)
            x1 = self.transform(x1)
            x2 = self.transform(x2)
 

        return (x0,x1), x2
  
class Dataset3D(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, x, dt, nInFrames = 3 , transform=None):
            'Initialization'
            self.x = x
            self.dt = dt
            self.transform = None
            self.convert_tensor = transforms.ToTensor()
            self.nInFrames = nInFrames

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.x)-self.nInFrames 

      def __getitem__(self, index):
            'Generates one sample of data'

            # Concatenate 3 frames to create 3D image
            for i in range(self.nInFrames):
                  x_temp = self.convert_tensor(genData.create_pendulum_image( self.x[index+i] ))
                 
                  if i == 0:  
                        input = x_temp
                  else :
                        input = torch.cat((input, x_temp), 0)
            
            
            
            
            # Select sample                


            out =self.convert_tensor( genData.create_pendulum_image( self.x[index + self.nInFrames ] ) )


            if self.transform:
                  input = self.transform(input)
                  out = self.transform(out)


            return input, out
      
class Dataset_decoder(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, x, transform=None):
            'Initialization'
            self.x = x
            self.transform = None
            self.convert_tensor = transforms.ToTensor()

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.x)

      def __getitem__(self, index):
            'Generates one sample of data'
            
            input = torch.tensor([self.x[index]])   
            out =self.convert_tensor( genData.create_pendulum_image( self.x[index ] ) )


            if self.transform:
                  input = self.transform(input)
                  out = self.transform(out)


            return input, out
            
def getLoader(X,  split = True, type = "Dataset3d"):

      if type == "Dataset":
            loadertype = Dataset
      if type == "Dataset3d":
            loadertype = Dataset3D
      if type == "Dataset_decoder":
            loadertype = Dataset_decoder

      if split:
     
            #split dataset 80-20 for training and validation

            train_x, val_x = train_test_split(X, test_size=0.2, shuffle=False)

            #create train and test dataloaders

            train_dataset = DataLoader( loadertype(train_x, 1/30), batch_size=32, shuffle=False)
            val_dataset = DataLoader( loadertype(val_x, 1/30), batch_size=32, shuffle=False)    

            return train_dataset, val_dataset, train_x, val_x 
      else :
            return DataLoader( loadertype(X, 1/30), batch_size=1, shuffle=False)
      

def get_template():

      convert_tensor = transforms.ToTensor()

      img = genData.create_pendulum_image( 0 ) 

      # show image

      plt.imshow(img)

      return convert_tensor( genData.create_pendulum_image( 0 ) )
      

if __name__ == "__main__":
    pass
