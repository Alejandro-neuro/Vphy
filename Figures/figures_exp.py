import Data.genData as genData
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os



def main():

    folder = "./Figures/figs_exp"
    n_images = 8

    if not os.path.exists(folder):
        os.makedirs(folder)

    for dyn_type in ["Scale"]:

        if dyn_type == "Motion":
                ImageGenerator = genData.create_pendulum_image
                t,a = genData.generateDynamics(0.5,(-0.5))
        if dyn_type == "Scale":
                ImageGenerator = genData.create_half_radius_circle_image
                t,a = genData.generateDynamics(max=10.0, min = -10.0, dt = 0.1)
        if dyn_type == "Intensity":
                ImageGenerator = genData.create_intensity_image
                t,a = genData.generateDynamics(1,0.2)


        for i in range(n_images):
            
            x = ImageGenerator(a[i*3], noise=False)

            x = (x*255).astype(np.uint8)

            image = Image.fromarray(x)
            
            if image.mode != 'RGB':
                image = image.convert('RGB') 

            name = folder+"/"+dyn_type+str(i)+".png"
            image.save(name)

        # Get the list of image files in the folder
    image_files = [f for f in os.listdir(folder) if  f.endswith(".png") and os.path.isfile(os.path.join(folder, f))]

    # Sort the image files
    image_files.sort()

    # Create a high-resolution subplot
    fig, axes = plt.subplots(3, n_images, figsize=(30, 10), facecolor='none', gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

    # Iterate over the image files and plot them in the subplot
    row_letters = ['A', 'B', 'C']
    for idx, image_file in enumerate(image_files):
        # Load the image
        image = plt.imread(os.path.join(folder, image_file))
        
        # Determine the row and column indices
        row_index = idx // n_images
        col_index = idx % n_images
        
        # Plot the image
        axes[row_index, col_index].imshow(image)
        axes[row_index, col_index].axis('off')
        
        # Add the letter label
        #axes[row_index, col_index].text(0.5, 0.5, row_letters[row_index], fontsize=20, color='red', ha='center', va='center')

    # Adjust layout
    plt.style.use('default')
    plt.tight_layout()
    name_sub = folder+"/fig_exp.png"
    plt.savefig(name_sub, dpi=300, transparent=True,bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

