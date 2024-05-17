# Plot the training and validation losses.
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import numpy as np
import os
from matplotlib import pyplot
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FFA500', '#800080', '#008080']

colors_dark = ['#00FFFF', '#FF1493', '#00FF00', '#FF4500', '#ADFF2F', '#FF00FF', '#1E90FF', '#FF69B4', '#20B2AA', '#FF8C00']


def plotMultiple( X,  xlabel, ylabel,title, name, styleDark = False, show = False, plot_type = 'plot' ):

    if(show):
        plt.ioff()
    
    plt.figure()
    # if(styleDark):
    #     plt.style.use('dark_background')
    # else:
    #     plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 10), dpi= 300)
    #plt.title(title,size=40)
    

    #create a funtion that iterates over the list of lists and plots each one
    for i,row in enumerate(X):
        x = row['x']
        y = row['y']
        try:
            color = row['color']
            
        except:
            if(styleDark):
                color = colors_dark[i]
            else:
                color = colors[i]
        #Set alpha default one if not defined
        try:
            alpha = row['alpha']
            #alpha = 1.0
        except:
            alpha = 1.0 
            
        try:
            linestyle = row['linestyle']
        except:
            linestyle = '-'
        
        try:
            linewidth = row['linewidth']
        except:
            linewidth = 3.0

        try:
            plot_type = row['plot_type']
        except:
            plot_type = 'plot'



        if plot_type == 'scatter':
            ax.scatter(x,y, color=color, label=f'{row["label"]}', alpha = alpha, s=5 )
        elif plot_type == 'hist':
            ax.hist(y, bins=100, color=color, label=f'{row["label"]}', alpha = alpha, linestyle = linestyle)
        else:
            ax.plot(x,y, color=color, linewidth =linewidth, label=f'{row["label"]}', alpha = alpha, linestyle = linestyle)
    
    #first_legend = plt.legend(fontsize="30", loc ="upper right")
    aa, = plt.plot(0,0, color='k', label='Train')
    aa1, =plt.plot(0,0, color='k', linestyle = '--', label='Test')
    lines = ax.get_lines()

    if True:
        first_legend = ax.legend(handles=lines[:-2], fontsize="30", loc ="lower right")
    else:

        first_legend = ax.legend(handles=lines[:-2], fontsize="30", loc ="upper right")
        ax.add_artist(first_legend)
        
        
        
        second_legend= plt.legend(handles=lines[-2:], fontsize="40", loc ="upper left")

    #set limits
    #plt.xlim(0, 100)
    #plt.ylim(0, 0.2)

    

    plt.rcParams['text.usetex'] = True
    #plt.legend(handles=[second_legend], loc='lower right')

    plt.grid(True, alpha=0.5)

    plt.xlabel(r'$ z_{real}$',size=50)
    plt.ylabel(r'$z_E$',size=50 )#, rotation=0)
    plt.tick_params(axis='y', which='major', pad=25)
    plt.rc('xtick', labelsize=35)
    plt.rc('ytick', labelsize=35)

    plt.rc('font', family='serif')
    
    folder = "./Figures/figs_init"
    name_plot = folder+"/"+name+".png"
    name_plot_eps = folder+"/"+name+".eps"
    plt.savefig(name_plot, dpi=300, transparent=True,bbox_inches='tight')
    plt.savefig(name_plot_eps, format='eps')
    if(show):
        plt.show()
    else:
        plt.close()




def plotMatrix(M,xlabel, ylabel,title, name, styleDark = False):

    plt.figure()
    if(styleDark):
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    fig, axarr = plt.subplots(figsize=(80, 10), dpi= 300)
    plt.title(title,size=40)
    plt.xlabel(xlabel,size=30)
    plt.ylabel(ylabel,size=30)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('font', family='serif')

    cmap = plt.cm.get_cmap('Greys_r', 256)

    axarr.set_xlim(0, M.shape[1])
    axarr.set_ylim(0, M.shape[0])
    
    
    img = axarr.imshow(M, cmap=cmap, interpolation='nearest')
    axarr.grid(which='major', color='purple', linestyle='-', linewidth=4)
    axarr.grid(which='minor', color='w', linestyle=':', linewidth=2)
    #axarr.grid(which='minor', color='purple', linestyle='-', linewidth=2)

    # Change major ticks to show every 20.
    axarr.xaxis.set_major_locator(MultipleLocator(10))
    axarr.yaxis.set_major_locator(MultipleLocator(10))

    # Change minor ticks to show every 5. (20/4 = 5)
    axarr.xaxis.set_minor_locator(AutoMinorLocator(4))
    axarr.yaxis.set_minor_locator(AutoMinorLocator(4))

    # Or if you want different settings for the grids:
    axarr.grid(which='minor', alpha=0.5)
    axarr.grid(which='major', alpha=0.8)

    axarr.set_xticklabels(axarr.get_xticks(), rotation = 45)


    cbar = plt.colorbar(img, ax=axarr, aspect=30)

    # Position the colorbar to the right of the plot
    #cbar.ax.yaxis.tick_right()
    cbar.ax.set_ylabel('Normalized voltage')

    
    plt.savefig(f'./plots/{name}.png')

    
    
    plt.show()

def plotAreas(x, GT = 0, parameter_name="" ):

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(5, 5))

    #plt.style.use('default')

    t = np.arange(0, x.shape[1])

    # Plot all vectors
    for i in range(x.shape[0]):
        
        ax.plot(t, x[i,:], color = "deepskyblue",  linewidth=1)
    
    #ax.plot(t, GT,'--', color = "deepskyblue",  linewidth=1)


    # Find the maximum value and its index among all vectors
    max_values = np.max(x, axis=0)
    min_values = np.min(x, axis=0)


    # Fill the area under the curve of the maximum value vector and under zero
    ax.fill_between(t, GT, max_values,  color='lightblue', alpha=0.5)
    ax.fill_between(t, min_values, GT,  color='lightblue', alpha=0.5)

    

    ax.axis('on')
    ax.grid(False)
    
    plt.rcParams['text.usetex'] = False
    plt.tight_layout()
    plt.tick_params(axis='x', colors='black')  # Set x-axis color to black
    plt.tick_params(axis='y', colors='black')
    plt.xscale('log')
    #ax.set_xscale('log')

    #set fotn size axis numbers
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=20)
    plt.rc('font', family='serif')


    folder = "./Figures/figs_init"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    name_plot = folder+"/"+parameter_name+".png"
    name_plot_eps = folder+"/"+parameter_name+".eps"
    plt.savefig(name_plot, dpi=300, transparent=True,bbox_inches='tight')
    plt.savefig(name_plot_eps, format='eps')
    plt.show()

    
