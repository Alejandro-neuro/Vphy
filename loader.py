import torch
import torch.nn as nn
import networkx as nx
import numpy as np

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score

from omegaconf import OmegaConf

import Data.genData as genData
from torchvision import transforms


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
        return len(self.x)-2

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
 

        return (x0,x1 ), x2
        
def getLoader(X):
     
      #split dataset 80-20 for training and validation

      train_x, val_x = train_test_split(X, test_size=0.2, shuffle=False)

      #create train and test dataloaders

      train_dataset = DataLoader( Dataset(train_x, 1/30), batch_size=32, shuffle=False)
      val_dataset = DataLoader( Dataset(val_x, 1/30), batch_size=32, shuffle=False)    

      return train_dataset, val_dataset, train_x, val_x 

if __name__ == "__main__":
    pass
