#!/usr/bin/env python
# coding: utf-8
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import matplotlib.pyplot as plt
import sys
from data.dataset_full import SituationalAwarenessDataset
    
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
        axs[i].imshow(image, cmap='gray')
        axs[i].axis('off')
        # axs[i].set_title(name)
    
    # Hide remaining axes
    for j in range(len(images), 21):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(name+ '.png')


args = argparse.ArgumentParser()
args.add_argument("--encoder", type=str, default='mobilenet_v2')
args.add_argument("--encoder-weights", type=str, default='imagenet')
args.add_argument("--classes", type=str, default='car')
args.add_argument("--activation", type=str, default='sigmoid')
args.add_argument("--device", type=str, default='cuda')
args.add_argument("--num-workers", type=int, default=4)

args.add_argument("--raw-data", type=str, default='/media/storage/raw_data_corrected')
args.add_argument("--return_rgb", action='store_true')
args.add_argument("--instseg_channels", type=int, default=2)
args.add_argument("--middle_only", action='store_false')

args.add_argument("--batch-size", type=int, default=16)


args.add_argument("--sensor-config-file", type=str, default='/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/sensor_config.ini')

args = args.parse_args()

# Lets look at data we have

dataset = SituationalAwarenessDataset(args.raw_data, args.sensor_config_file, "cbdr10-23", args)
# import ipdb; ipdb.set_trace()

for i in range(0, len(dataset), 25):
    print(i)
    final_concat_image, label_mask_image = dataset[i] # get some sample
    np.save("test_input_output/example_input_" +str(i)+".npy", final_concat_image.numpy())
    np.save("test_input_output/example_output_" +str(i)+".npy", label_mask_image.numpy())
    visualize_all(final_concat_image.numpy(), "test_input_output/input_" + str(i))
    visualize_all(label_mask_image.numpy(), "test_input_output/output_" + str(i))