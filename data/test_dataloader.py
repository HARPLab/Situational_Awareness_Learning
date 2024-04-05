#!/usr/bin/env python
# coding: utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import matplotlib.pyplot as plt
import sys
from data.dataset_full_scipy import SituationalAwarenessDataset
    
import torch
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

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


# Lets look at data we have
sensor_config_file = "/home/pranaygu-local/Situational_Awareness_Learning/sensor_config.ini"
raw_data = "/scratch/pranaygu/raw_data"

sitawdata = SituationalAwarenessDataset(raw_data, sensor_config_file, "cbdr10-23")
# import ipdb; ipdb.set_trace()

for i in range(0, 100, 25):
    print(i)
    final_concat_image, label_mask_image = sitawdata[i] # get some sample
    # np.save("test_input_output/example_input_" +str(i)+".npy", final_concat_image.numpy())
    # np.save("test_input_output/example_output_" +str(i)+".npy", label_mask_image.numpy())
    visualize_all(final_concat_image.numpy(), "test_input_output/input_" + str(i))
    visualize_all(label_mask_image.numpy(), "test_input_output/output_" + str(i))