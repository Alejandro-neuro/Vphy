import torch
import torch.nn as nn
import networkx as nx
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score

from omegaconf import OmegaConf



def gen_Dataset(x, type, window=20, overlap=0.5):
    dataset = []
    

    for i in range( x.shape[1] - window-1 ):

        datax = x[:,i:i+window]
        label = x[:,i+window+1]

        adj = generateAdjacencyMatrix(datax,type)
        G = nx.DiGraph(adj)
        data1 = from_networkx(G)  

        data = data1.clone()
        data['x']=torch.from_numpy(datax).float()
        data['y']=torch.from_numpy(label).float()        
        data['weight']=data['weight'].float()

        dataset.append(data)

        i = i + int(window * overlap)

    return dataset

def timeSeries2Dataset(timeseries, type, window=20, overlap=0.5):

    data = timeseries

    len_data = data.shape[1]

    dataset={
        "trn": {
        "data": None,
        "labels": None
        },
        "val":  {
        "data": None,
        "labels": None
        },
        "tst":  {
        "data": None,
        "labels": None
        }
    }

    cfg = OmegaConf.load("config.yaml")

    train_size = cfg.dataset.train_size
    val_size = cfg.dataset.val_size
    test_size = cfg.dataset.test_size

    X_train = data[:,:int(train_size*len_data)]
    X_val = data[:,int(train_size*len_data):int((train_size+val_size)*len_data)]
    X_test = data[:,int((train_size+val_size)*len_data):]   

    print("X_train shape: ", X_train.shape)
    print("X_val shape: ", X_val.shape)
    print("X_test shape: ", X_test.shape)

    dataset['trn'] = gen_Dataset(X_train, type, window=window, overlap=0.5)
    dataset['val'] = gen_Dataset(X_val, type, window=window, overlap=0.5)
    dataset['tst'] = gen_Dataset(X_test, type, window=window, overlap=0.5)
    
    return dataset




def create_partitions(dataset):
    
    train_loader = DataLoader(dataset['trn'], batch_size=64, shuffle=True)
    val_loader = DataLoader( dataset['val'], batch_size=64, shuffle=False)
    test_loader = DataLoader( dataset['tst'], batch_size=64, shuffle=False)
 
    return train_loader,val_loader,test_loader

#create a function that returns adjacency matrix and based on the input string "type" and returns pearson correlation
#or mutual information
def generateAdjacencyMatrix(x,type):
    
    if(type == 'pearson'): 
    #calculate adjacency matrix with pearson correlation
        adj = np.zeros((x.shape[0],x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                adj[i,j] = np.corrcoef(x[i,:],x[j,:])[0,1] #calculate pearson correlation between nodes i and j
                adj[j,i] = adj[i,j] #make the matrix symmetric

        return adj

    
    if(type == 'mutual_info'):
        adj = np.zeros((x.shape[0],x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                adj[i,j] = mutual_info_score(x[i,:],x[j,:]) #calculate mutual information between nodes i and j
                adj[j,i] = adj[i,j] #make the matrix symmetric  
        return adj
    
    #calculate adjacency matrix with cosine similarity
    if(type == 'cosine'):
        adj = np.zeros((x.shape[0],x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                adj[i,j] = np.dot(x[i,:],x[j,:]) / (np.linalg.norm(x[i,:]) * np.linalg.norm(x[j,:]))
                adj[j,i] = adj[i,j] #make the matrix symmetric
        return adj   

    return adj
        

def generateLoaders( timeseries, type='pearson', window=20, overlap=0.5):

    dataset = timeSeries2Dataset(timeseries, type, window, overlap)

    train_loader,val_loader,test_loader = create_partitions(dataset)

    return train_loader,val_loader,test_loader

if __name__ == "__main__":
    generateLoaders()

