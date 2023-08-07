# Plot the training and validation losses.
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FFA500', '#800080', '#008080']

colors_dark = ['#00FFFF', '#FF1493', '#00FF00', '#FF4500', '#ADFF2F', '#FF00FF', '#1E90FF', '#FF69B4', '#20B2AA', '#FF8C00']


def plotMultiple( X,  xlabel, ylabel,title, name, styleDark = False ):
    
    plt.figure()
    if(styleDark):
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    fig, axarr = plt.subplots(figsize=(20, 10), dpi= 80)
    plt.title(title,size=40)
    plt.xlabel(xlabel,size=30)
    plt.ylabel(ylabel,size=30)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('font', family='serif')

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


    

        plt.plot(x,y, color=color, linewidth =3, label=f'{row["label"]}' )
    
    plt.legend(fontsize="20", loc ="upper left")
    plt.savefig(f'./plots/{name}.png')
    
    plt.show()

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
    
