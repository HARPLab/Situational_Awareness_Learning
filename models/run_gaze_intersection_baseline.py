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
from custom_train import Classical_Baseline_Epoch, Classical_Baseline_VizEpoch
from dice_loss import DiceLoss   
from custom_metrics import gaze_intersection_object_level_accuracy
from gaze_intersection_baseline import GazeIntersection

import albumentations as albu

   
def main(args):
    ENCODER = args.encoder
    if args.seg_mode == 'multilabel':
        CLASSES = ['aware', 'not_aware']
        class_weights = [1, args.unaware_classwt]
    elif args.seg_mode == 'multiclass':
        CLASSES = ['aware', 'not_aware', 'bg']
        class_weights = [1, args.unaware_classwt, args.bg_classwt]
    ACTIVATION = args.activation # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = args.device

    num_images_per_sample = 1 if not args.middle_andsides else 3
    in_channels = num_images_per_sample*(3*(args.use_rgb) + args.instseg_channels + 1)

    # create segmentation model with pretrained encoder
    model = GazeIntersection()
    episode_list = sorted(os.listdir(args.raw_data), reverse=False)
    num_val_episodes = args.num_val_episodes
    train_batch_size = args.batch_size
    
    # set random seed and shuffle dataset
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    np.random.shuffle(episode_list)
    val_episodes = ["cbdr8-54" , "cbdr9-23", "cbdr6-41", "abd-21"]
    wandb_run_name = "%s_m%s_rgb%s_seg%d_mode_%s_sh%.1f@%.1f_gf%s_sample_%s_wus%s_ioc%s" % (ENCODER, args.middle_andsides,
                            args.use_rgb, args.instseg_channels, args.seg_mode,
                            args.secs_of_history, args.history_sample_rate,
                            args.gaze_fade, args.sample_clicks, args.weighted_unaware_sampling, args.ignore_oldclicks) + args.run_name

    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            entity="harplab-SA",
            project="DRVR-SA",        
            # track hyperparameters and run metadata
            config= {"args":args,
            "val_episodes":val_episodes},
            # set a name for the run  
            name=wandb_run_name      
        )


    #region: Load data

    valid_data = []
    concat_val_sample_weights = []
    for ep in val_episodes:
        dataset = SituationalAwarenessDataset(args.raw_data, args.sensor_config_file, ep, args)
        valid_data.append(dataset)
        concat_val_sample_weights += dataset.get_sample_weights()
    valid_dataset = torch.utils.data.ConcatDataset(valid_data)

        
    valid_loader = DataLoader(valid_dataset, batch_size=train_batch_size, shuffle=False, num_workers=args.num_workers)
    #endregion
    

    # note: metric computations expect activated predictions
    # we do activations in the loss function, so custom_train implements it prior to metric computation
    metrics = [
        # smp_utils.metrics.IoU(threshold=0.5),
        # IoU(threshold=0.5),
        gaze_intersection_object_level_accuracy(threshold=0.5, remove_small_objects=args.remove_small_objects), 
    ]


    valid_epoch = Classical_Baseline_Epoch(
        model, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    valid_visualization_epoch = Classical_Baseline_VizEpoch(
        model,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
        args=args
    )

    
    max_score = 0

    # initial visualization to make sure inputs are correct
    if args.wandb:
        valid_viz_logs = valid_visualization_epoch.run(valid_data[0])
        for j, fig in enumerate(valid_viz_logs):
            wandb.log({"val_visualizations_{}".format('init'): fig})

    

    # train model
    for cur_epoch in range(0, args.num_epochs):
        
        print('\nEpoch: {}'.format(cur_epoch))
        valid_logs, val_pred, val_gt = valid_epoch.run(valid_loader)
        
        if args.wandb:
            for k in valid_logs:
                wandb.log({"valid_"+k: valid_logs[k]})
            
            if not args.dont_log_images:
                # train_viz_idx = np.random.choice(range(len(train_data)), 1)
                train_viz_idx = 0
                val_viz_idx = 0
                for idx in range(len(valid_data)):
                    valid_viz_logs = valid_visualization_epoch.run(valid_data[idx])

                    for j, fig in enumerate(valid_viz_logs):
                        wandb.log({"val_visualizations_{}".format(cur_epoch): fig})
            
            
            if 'object_level_accuracy' in valid_logs:
                np.save(wandb.run.dir + '/best_val_preds.npy', val_pred)
                np.save(wandb.run.dir + '/best_val_gt.npy', val_gt) 


        # do something (save model, change lr, etc.)            

    if args.wandb:
        wandb.finish()

    # # load best saved checkpoint
    # best_model = torch.load('./best_model.pth')

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # model params
    args.add_argument("--architecture", choices=['fpn', 'unet', 'deeplabv3'], default='fpn')
    args.add_argument("--encoder", choices=['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2', 'efficientnet-b0'], default='mobilenet_v2')
    args.add_argument("--encoder-weights", choices=['imagenet', 'swsl', 'ssl', 'instagram', None], default=None)
    # args.add_argument("--classes", type=str, default='car')
    # new dice loss does activation in the loss function
    args.add_argument("--activation", choices=[None], default=None)    
    args.add_argument("--seg-mode", choices=['binary', 'multiclass', 'multilabel'], default='multiclass')

    # data set config params
    args.add_argument("--sensor-config-file", type=str, default='sensor_config.ini')
    args.add_argument("--raw-data", type=str, default='/media/storage/raw_data_corrected')
    args.add_argument("--use-rgb", action='store_true')
    args.add_argument("--instseg-channels", type=int, default=1)
    args.add_argument("--middle-andsides", action='store_true')
    args.add_argument("--secs-of-history", type=float, default=5.0)
    args.add_argument("--history-sample-rate", type=float, default=4.0)
    args.add_argument("--gaze-gaussian-sigma", type=float, default=5.0)
    args.add_argument("--gaze-fade", action='store_true')
    args.add_argument("--gaze-format", choices=['dot', 'blob'], default='blob')
    args.add_argument("--lr-decay-epochstep", type=int, default=10)
    args.add_argument("--lr-decay-factor", type=int, default=10)
    args.add_argument("--sample-clicks", choices=['post_click', 'pre_excl', 'both', ''], 
                      default='', help="Empty string -> sample everything")
    args.add_argument("--ignore-oldclicks", action='store_true')
    args.add_argument("--weighted-unaware-sampling", action='store_true', help="equally sample images with atleast one unaware obj and images with no unaware obj")
    args.add_argument("--pre-clicks-excl-time", type=float, default=1.0, help="seconds before click to exclude for reaction time")
    args.add_argument("--unaware-classwt", type=float, default=1.0)
    args.add_argument("--bg-classwt", type=float, default=1e-5)
    args.add_argument("--aware-threshold", type=float, default=0.5)
    args.add_argument("--unaware-threshold", type=float, default=0.5)
    args.add_argument("--remove-small-objects", action='store_true')
    args.add_argument("--synthetic-gaze", action='store_true')
    args.add_argument("--gaze-points-per-frame", type=int, default=25)



    # training params
    args.add_argument("--device", type=str, default='cuda')
    args.add_argument("--random-seed", type=int, default=999)
    args.add_argument("--num-workers", type=int, default=12)
    args.add_argument("--batch-size", type=int, default=16)
    args.add_argument("--num-val-episodes", type=int, default=5)
    args.add_argument("--num-epochs", type=int, default=20)
    args.add_argument("--lr", type=float, default=0.0001)
    args.add_argument("--wandb", action='store_false')
    args.add_argument("--dont-log-images", action='store_true')
    args.add_argument("--image-save-freq", type=int, default=150)
    args.add_argument("--unfix-valset", action='store_true')    
    args.add_argument("--resume-path", type=str, default="")
    
    args.add_argument("--run-name", type=str, default="")    
    args = args.parse_args()

    main(args)    