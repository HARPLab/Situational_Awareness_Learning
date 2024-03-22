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

# Lets look at data we have
awareness_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/results/exp_allan_51-awdata.json"
images_dir = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_allan-51_02_20_2024_17_21_58/images"
awareness_df = pd.read_json(awareness_parse_file, orient='index')
sensor_config_file = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/sensor_config.ini"
recording_path = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/recording_files/exp_allan-51_02_20_2024_17_21_58.rec"

sitawdata = SituationalAwarenessDataset(recording_path, images_dir, awareness_df, sensor_config_file, gaussian_sigma = 40.0)


rgb_image, instance_seg_image, gaze_heatmap, rgb_left_image, instance_seg_left_image, gaze_heatmap_left, rgb_right_image, instance_seg_right_image, gaze_heatmap_right, gaze_heatmap, validity = sitawdata[1525] # get some sample
visualize(
    image=rgb_image, 
    instance_mask=instance_seg_image,
)



# ## Create model and train
import torch
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils



# encoder = 'se_resnext50_32x4d'
# encoder_weights = 'imagenet'

ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['car']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)


# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])



# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)


# train model for 40 epochs

max_score = 0

for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')


# ## Test best saved model


# load best saved checkpoint
best_model = torch.load('./best_model.pth')

# create test dataset
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)


# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)


# ## Visualize predictions

# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir, 
    classes=CLASSES,
)


for i in range(5):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )

