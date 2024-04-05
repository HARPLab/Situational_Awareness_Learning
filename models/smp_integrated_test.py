#!/usr/bin/env python
# coding: utf-8


# Install required libs
#!pip install -U segmentation-models-pytorch albumentations --user 



#!pip uninstall -y segmentation-models-pytorch


# ## Loading data

# For this example we will use **CamVid** dataset. It is a set of:
#  - **train** images + segmentation masks
#  - **validation** images + segmentation masks
#  - **test** images + segmentation masks
#  
# All images have 320 pixels height and 480 pixels width.
# For more inforamtion about dataset visit http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ### Dataloader
# 
# Writing helper class for data extraction, tranformation and preprocessing  
# https://pytorch.org/docs/stable/data

import pandas as pd
import matplotlib.pyplot as plt
from data.dataset_full import SituationalAwarenessDataset
    

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

def visualize_all(**images):
    fig, axs = plt.subplots(4, 3, figsize=(10, 10))
    axs = axs.flatten()
    
    for i, (name, image) in enumerate(images.items()):
        axs[i].imshow(image)
        axs[i].axis('off')
        axs[i].set_title(name)
    
    # Hide remaining axes
    for j in range(len(images), 12):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Lets look at data we have
# sensor_config_file = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/sensor_config.ini"
# raw_data = "/home/srkhuran-local/raw_data"

# sitawdata = SituationalAwarenessDataset(raw_data, sensor_config_file, "cbdr10-36")

# final_concat_image, label_mask_image, validity = sitawdata[51] # get some sample
# visualize_all(
#     rgb_mid=rgb_image, 
#     instance_seg_mid=instance_seg_image,
#     gaze_mid=gaze_heatmap,
#     rgb_left=rgb_left_image, 
#     instance_seg_left=instance_seg_left_image,
#     gaze_left=gaze_heatmap_left,
#     rgb_right=rgb_right_image, 
#     instance_seg_right=instance_seg_right_image,
#     gaze_right=gaze_heatmap_right,
#     label=label_mask_image
# )



# ## Create model and train
import torch
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils
import argparse 


# encoder = 'se_resnext50_32x4d'
# encoder_weights = 'imagenet'
# add argparse for encoder and encoder_weights

args = argparse.ArgumentParser()
args.add_argument("--encoder", type=str, default='mobilenet_v2')
args.add_argument("--encoder-weights", type=str, default='imagenet')
args.add_argument("--classes", type=str, default='car')
args.add_argument("--activation", type=str, default='sigmoid')
args.add_argument("--device", type=str, default='cuda')
args.add_argument("--num-workers", type=int, default=0)

args.add_argument("--raw-data", type=str, default='/scratch/pranaygu/raw_data')
args.add_argument("--return_rgb", action='store_true')
args.add_argument("--instseg_channels", type=int, default=2)
args.add_argument("--middle_only", action='store_false')


args.add_argument("--sensor-config-file", type=str, default='/home/pranaygu-local/Situational_Awareness_Learning/sensor_config.ini')

args = args.parse_args()

ENCODER = args.encoder
ENCODER_WEIGHTS = args.encoder_weights
CLASSES = ['aware', 'not_aware']
ACTIVATION = args.activation # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = args.device

num_images_per_sample = 1 if args.middle_only else 3
in_channels = num_images_per_sample*(3*(args.return_rgb) + args.instseg_channels + 1)

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=args.encoder, 
    encoder_weights=args.encoder_weights, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
    in_channels=in_channels,
)

#preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

episode_list = os.listdir(args.raw_data)
# train_episodes, val_episodes, test_episodes = data.split_train_val_test(episode_list, 0.8, 0.1, 0.1) # NOTE: DOUBLE CHECK THIS
train_episodes = [episode_list[0]]
print(train_episodes)
train_data = []
for ep in train_episodes:
    dataset = SituationalAwarenessDataset(args.raw_data, args.sensor_config_file, ep, args)
    train_data.append(dataset)
train_dataset = torch.utils.data.ConcatDataset(train_data)

# valid_data = []
# for ep in val_episodes:
#     dataset = SituationalAwarenessDataset(raw_data, sensor_config_file, ep)
#     valid_data.append(dataset)
# valid_dataset = torch.utils.data.ConcatDataset(valid_data)

# test_data = []
# for ep in test_episodes:
#     dataset = SituationalAwarenessDataset(raw_data, sensor_config_file, ep)
#     test_data.append(dataset)
# test_dataset = torch.utils.data.ConcatDataset(test_data)


# train_dataset = SituationalAwarenessDataset(raw_data, sensor_config_file, train_episodes) 
# valid_dataset = SituationalAwarenessDataset(raw_data, sensor_config_file, val_episodes)
# test_dataset = SituationalAwarenessDataset(raw_data, sensor_config_file, test_episodes)


# full_dataset = SituationalAwarenessDataset(raw_data, sensor_config_file) # NOTE: may need to take in run number as input and create seperate dataset for each run (concatenate them for training)
# train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.1, 0.1])
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
# valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
#valid_dataset = SituationalAwarenessDataset()


# train_dataset = Dataset(
#     x_train_dir, 
#     y_train_dir, 
#     augmentation=get_training_augmentation(), 
#     preprocessing=get_preprocessing(preprocessing_fn),
#     classes=CLASSES,
# )

# valid_dataset = Dataset(
#     x_valid_dir, 
#     y_valid_dir, 
#     augmentation=get_validation_augmentation(), 
#     preprocessing=get_preprocessing(preprocessing_fn),
#     classes=CLASSES,
# )


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=args.num_workers)
# valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)


# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp_utils.losses.DiceLoss()
metrics = [
    smp_utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])



# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp_utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

# valid_epoch = smp_utils.train.ValidEpoch(
#     model, 
#     loss=loss, 
#     metrics=metrics, 
#     device=DEVICE,
#     verbose=True,
# )


# train model for 40 epochs
max_score = 0

for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    # valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < train_logs['iou_score']:
        max_score = train_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')


# ## Test best saved model


# load best saved checkpoint
best_model = torch.load('./best_model.pth')


# test_dataloader = DataLoader(test_dataset)


# # evaluate model on test set
# test_epoch = smp_utils.train.ValidEpoch(
#     model=best_model,
#     loss=loss,
#     metrics=metrics,
#     device=DEVICE,
# )

# logs = test_epoch.run(test_dataloader)


# # ## Visualize predictions

# # test dataset without transformations for image visualization
# test_dataset_vis = Dataset(
#     x_test_dir, y_test_dir, 
#     classes=CLASSES,
# )


# for i in range(5):
#     n = np.random.choice(len(test_dataset))
    
#     image_vis = test_dataset_vis[n][0].astype('uint8')
#     image, gt_mask = test_dataset[n]
    
#     gt_mask = gt_mask.squeeze()
    
#     x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
#     pr_mask = best_model.predict(x_tensor)
#     pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
#     visualize(
#         image=image_vis, 
#         ground_truth_mask=gt_mask, 
#         predicted_mask=pr_mask
#     )

