import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from Data import genData as genData
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from src import train
from src import loader
from src.models import model as mainmodel
from src import custom_plots as cp


def main():


    for dyn_type in ["Motion","Scale","Intensity"]:

        print("Dynamics type: ", dyn_type)

        if dyn_type == "Motion":
            t,a = genData.generateDynamics(max=0.5, min = -0.5)
        if dyn_type == "Scale":
            t,a = genData.generateDynamics(max=2, min = 0)
        if dyn_type == "Intensity":
            t,a = genData.generateDynamics(max=1, min = 0.2)

        sr = 20
        dt_sim = t[1]-t[0]
        dt = dt_sim*sr
        train_dataloader, test_dataloader, train_x, val_x  = loader.getLoader(a,type=dyn_type, split = True,  dt=dt_sim, nInFrames = 5,sr = sr ,  noise=False, shapeType='simple')
        

        a = []
        b = []

        best_a = []	
        best_b = []

        plt.figure()

        
        for inits in [ -10.0, -5.0, -1.0,0.0, 1.0, 5.0, 10.0]:
        
            latentEncoder = mainmodel.EndPhys(dt = dt,  
                                            pmodel = "Damped_oscillation",
                                            init_phys = inits, 
                                            initw=True)
            
            latentEncoder, log  = train.train(latentEncoder, 
                                            train_dataloader, 
                                            test_dataloader,                                 
                                            loss_name='latent_loss')
            
            best_a.append(latentEncoder.pModel.alpha[0].detach().cpu().numpy().item())
            best_b.append(latentEncoder.pModel.beta[0].detach().cpu().numpy().item())
            
            a.append( [element["alpha"] for element in log  ])
            b.append( [element["beta"] for element in log  ] )

        a = np.array(a)
        b = np.array(b)
        cp.plotAreas(a, 3.99, dyn_type+"_gamma1")
        cp.plotAreas(b, 0.04, dyn_type+"_gamma2")

        best_a = np.array(best_a)
        best_b = np.array(best_b)

        print("Best alpha: ", best_a.mean()) 
        print("Best beta: ", best_b.mean())
        print("Best alpha std: ", best_a.std())
        print("Best beta std: ", best_b.std())

    
if __name__ == "__main__":
    main()
