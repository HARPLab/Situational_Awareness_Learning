import numpy as np
import matplotlib.pyplot as plt

def visualize_all(images, name):
    fig, axs = plt.subplots(7, 3, figsize=(10, 10))
    axs = axs.flatten()
    
    for i, (image) in enumerate(images):
        axs[i].imshow(image)
        axs[i].axis('off')
        # axs[i].set_title(name)
    
    # Hide remaining axes
    for j in range(len(images), 21):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(name+ '.png')

a = np.load('example_input.npy')
b = np.load('example_output.npy')
visualize_all(a, "input")
visualize_all(b, "output")