
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os






def main():

    folder = "./Figures/test"
    

    if not os.path.exists(folder):
        os.makedirs(folder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device",device)

    x = np.arange(0,100,0.1)
    y = np.sin(x)



    plt.plot(x,y)

    name_sub = folder+"/fig_exp.png"

    plt.savefig(name_sub, dpi=300, transparent=True,bbox_inches='tight')

if __name__ == "__main__":
    main()  