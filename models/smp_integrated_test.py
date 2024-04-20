import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os.path
home_folder = os.path.expanduser('~')
sys.path.insert(0, os.path.join(home_folder, 'Situational_Awareness_Learning'))
from data.dataset_full import SituationalAwarenessDataset
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils
import argparse 
import wandb
sys.path.insert(0, './models/')
from custom_train import *
   

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



def main(args):
    ENCODER = args.encoder
    CLASSES = ['aware', 'not_aware']
    ACTIVATION = args.activation # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = args.device

    num_images_per_sample = 1 if args.middle_andsides else 3
    in_channels = num_images_per_sample*(3*(args.use_rgb) + args.instseg_channels + 1)
    train_batch_size = args.batch_size

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=args.encoder, 
        encoder_weights=args.encoder_weights, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
        in_channels=in_channels,
    )

    #preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # episode_list = sorted(os.listdir(args.raw_data), reverse=True)
    episode_list = sorted(os.listdir(args.raw_data), reverse=False)
    num_val_episodes = args.num_val_episodes
    # train_episodes, val_episodes, test_episodes = data.split_train_val_test(episode_list, 0.8, 0.1, 0.1) # NOTE: DOUBLE CHECK THIS
    
    # set random seed and shuffle dataset
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    np.random.shuffle(episode_list)
    train_episodes = episode_list[:-num_val_episodes]   
    # train_episodes = [episode_list[0]] 
    print("Train routes:", train_episodes)
    val_episodes = episode_list[-num_val_episodes:]
    # val_episodes = [episode_list[0]] 
    print("Val routes:", val_episodes)

    wandb_run_name = "%s_m%s_rgb%s_seg%d_sh%.1f@%.1f_g%.1f_gf%s_sample_%s" % (ENCODER, args.middle_andsides, args.use_rgb,
            args.instseg_channels, args.secs_of_history, 
            args.history_sample_rate, args.gaze_gaussian_sigma, args.gaze_fade, args.sample_clicks) + args.run_name

    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            entity="harplab-SA",
            project="DRVR-SA",        
            # track hyperparameters and run metadata
            config= {"args":args,
            "train_episodes":train_episodes,
            "val_episodes":val_episodes},
            # set a name for the run  
            name=wandb_run_name      
        )

    train_data = []
    concat_sample_weights = []
    for ep in train_episodes:
        dataset = SituationalAwarenessDataset(args.raw_data, args.sensor_config_file, ep, args)
        train_data.append(dataset)
        concat_sample_weights += dataset.get_sample_weights()
    train_dataset = torch.utils.data.ConcatDataset(train_data)


    valid_data = []
    concat_val_sample_weights = []
    for ep in val_episodes:
        dataset = SituationalAwarenessDataset(args.raw_data, args.sensor_config_file, ep, args)
        valid_data.append(dataset)
        concat_val_sample_weights += dataset.get_sample_weights()
    valid_dataset = torch.utils.data.ConcatDataset(valid_data)

    if args.weighted_unaware_sampling:
        weighted_sampler = WeightedRandomSampler(weights=concat_sample_weights,
                                                num_samples=len(train_dataset),
                                                replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=weighted_sampler, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=args.num_workers)
    
    weighted_sampler = WeightedRandomSampler(weights=concat_val_sample_weights, num_samples=len(valid_dataset), replacement=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=train_batch_size, shuffle=False, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=train_batch_size, sampler=weighted_sampler, num_workers=args.num_workers)


    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    loss = smp_utils.losses.DiceLoss()
    metrics = [
        smp_utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=args.lr),
    ])



    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    # train model for 40 epochs
    max_score = 0

    for i in range(0, args.num_epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        for k in train_logs:
            wandb.log({"train_"+k: train_logs[k]})
        for k in valid_logs:
            wandb.log({"valid_"+k: valid_logs[k]})
        # do something (save model, change lr, etc.)
        if max_score < train_logs['iou_score']:
            max_score = train_logs['iou_score']
            if args.wandb:
                torch.save(model,
                    os.path.join(wandb.run.dir, './best_model_%s.pth' 
                    % wandb_run_name))
            else:
                torch.save(model, 
                    './best_model_%s.pth' 
                    % wandb_run_name)

            
        if i > 0 and i % args.lr_decay_epochstep == 0:
            optimizer.param_groups[0]['lr'] /= 10
            print('Decimating decoder learning rate to %f' % optimizer.param_groups[0]['lr'])


    # ## Test best saved model
    if args.wandb:
        wandb.finish()

    # # load best saved checkpoint
    # best_model = torch.load('./best_model.pth')

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # model params
    args.add_argument("--encoder", type=str, default='mobilenet_v2')
    args.add_argument("--encoder-weights", type=str, default='imagenet')
    # args.add_argument("--classes", type=str, default='car')
    args.add_argument("--activation", type=str, default='sigmoid')
    
    # data set config params
    args.add_argument("--sensor-config-file", type=str, default='sensor_config.ini')
    args.add_argument("--raw-data", type=str, default='/media/storage/raw_data_corrected')
    args.add_argument("--use-rgb", action='store_true')
    args.add_argument("--instseg-channels", type=int, default=2)
    args.add_argument("--middle-andsides", action='store_false')
    args.add_argument("--secs-of-history", type=float, default=5.0)
    args.add_argument("--history-sample-rate", type=float, default=4.0)
    args.add_argument("--gaze-gaussian-sigma", type=float, default=10.0)
    args.add_argument("--gaze-fade", action='store_true')
    args.add_argument("--lr-decay-epochstep", type=int, default=10)
    args.add_argument("--sample-clicks", choices=['post_click', 'pre_excl', 'both', ''], 
                      default='', help="Empty string -> sample everything")
    args.add_argument("--ignore-oldclicks", action='store_true')
    # empty string is sample everything
    args.add_argument("--weighted-unaware-sampling", action='store_true')

    # training params
    args.add_argument("--device", type=str, default='cuda')
    args.add_argument("--random-seed", type=int, default=999)
    args.add_argument("--num-workers", type=int, default=12)
    args.add_argument("--batch-size", type=int, default=16)
    args.add_argument("--num-val-episodes", type=int, default=5)
    args.add_argument("--num-epochs", type=int, default=40)
    args.add_argument("--lr", type=float, default=0.0001)
    args.add_argument("--wandb", action='store_false')
    
    args.add_argument("--run-name", type=str, default="")    
    args = args.parse_args()

    main(args)    