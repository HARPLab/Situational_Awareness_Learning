import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os.path
home_folder = os.path.expanduser('~')
sys.path.insert(0, os.path.join(home_folder, 'Situational_Awareness_Learning'))
from data.dataset_full import SituationalAwarenessDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils
import argparse 
import wandb
from PIL import Image
import cv2


def visualize_one(images, name):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    # axs = axs.flatten()
    axs.imshow(images)
    axs.axis('off')
        # axs[i].set_title(name)
    
    # Hide remaining axes
    # for j in range(len(images), 21):
    #     axs.axis('off')
    
    plt.tight_layout()
    plt.savefig(name+ '.png')
    plt.close(fig)

def viz_inputs_with_gaze_overlaid(img_inputs, rgb, gt, pr, name):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    gaze_heatmap = np.array(img_inputs[-1])[4:-4, :]*255

    # Plot the images
    axes[0, 0].imshow(img_inputs[0])
    axes[0, 0].set_title('Instance Segmentation (R channel)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_inputs[1])
    axes[0, 1].set_title('Instance Segmentation (G channel)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(gaze_heatmap)
    axes[1, 1].set_title('Gaze heatmap')
    axes[0, 2].axis('off')

    # fig1 = plt.figure()
    # foo = cv2.merge((gaze_heatmap, np.zeros_like(gaze_heatmap), np.zeros_like(gaze_heatmap)))
    # ax1 = fig1.add_subplot(1,1,1)
    # ax1.imshow(gaze_heatmap)
    # fig1.savefig(name + '_gaze.png')

    axes[1, 0].imshow(cv2.addWeighted(np.array(rgb), 1, 
                                      255*cv2.merge((gaze_heatmap, np.zeros_like(gaze_heatmap), np.zeros_like(gaze_heatmap))), 
                                      0.5, 0))    
    # print(blended_array.shape)
    # axes[1, 0].imshow()
    axes[1, 0].set_title('RGB + gaze heatmap')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gt)
    axes[1, 1].set_title("Ground Truth")
    axes[1, 1].axis('off')
    axes[1, 2].imshow(pr)
    axes[1, 2].set_title("Prediction")
    axes[1, 2].axis('off')

    
    plt.tight_layout()
    plt.savefig(name+ '.png')
    plt.close(fig)

def visualize_all(images, name):
    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    axs = axs.flatten()
    
    for i, (image) in enumerate(images):
        axs[i].imshow(image)
        axs[i].axis('off')
        # axs[i].set_title(name)
    
    # Hide remaining axes
    for j in range(len(images), 2):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(name+ '.png')
    plt.close(fig)

# ## Create model and train
# encoder = 'se_resnext50_32x4d'
# encoder_weights = 'imagenet'
# add argparse for encoder and encoder_weights

def save_outputs(train_data, episodes, best_model, wandb_run_name, DEVICE, type):
    for i in range(len(train_data)):
        if os.path.exists(os.path.join(args.viz_output_path, wandb_run_name, episodes[i])) == False:
            os.makedirs(os.path.join(args.viz_output_path, wandb_run_name, episodes[i]))
            
        if os.path.exists(os.path.join(args.viz_output_path, wandb_run_name, episodes[i], type)) == False:
            os.makedirs(os.path.join(args.viz_output_path, wandb_run_name, episodes[i], type))

        for j in range(len(train_data[i])):
            if j%10 == 0:
                print(i, j)
                image, gt_mask, mask = train_data[i][j]
                frame_num = train_data[i].index_mapping[j]
                try:
                    rgb_image = Image.open(train_data[i].images_dir / 'rgb_output' / ('%.6d.png' % frame_num)).convert('RGB')
                except:
                    print(train_data[i].images_dir / 'rgb_output' / ('%.6d.png' % frame_num))
                    continue

                im0 = Image.fromarray(np.uint8(image.numpy()[0]*255))
                im1 = Image.fromarray(np.uint8(image.numpy()[1]*255))
                im2 = Image.fromarray(np.uint8(image.numpy()[2]*255))
                # visualize_all([im0, im1, im2], os.path.kjoin(args.viz_output_path, "inputs_train_" + str(n)))
                # gt_mask = gt_mask.squeeze()
                # pr_mask = best_model.predict(image.to(DEVICE).unsqueeze(0))
                # print(pr_mask.shape)
                # pr_mask = (pr_mask.squeeze().cpu().numpy().round())
                # import ipdb; ipdb.set_trace()
                # pr_fin = np.zeros([pr_mask.shape[1], pr_mask.shape[2], 3])
                # pr_fin[pr_mask[0] > pr_mask[1]] = np.array([0, 255, 0])
                # pr_fin[pr_mask[0] <= pr_mask[1]] = np.array([255, 0, 0])
                # pr_fin[np.sum(pr_mask, axis=0) == 0] = np.array([0, 0, 0])
                
                # bg_pixels = np.sum(pr_mask, axis=0) == 0
                # pr_fin = np.argmax(pr_mask, axis = 0)
                # pr_fin[bg_pixels] = -1
                # pr_fin += 1
                
                gt_mask = (gt_mask.cpu().numpy().round())
                gt_fin = np.zeros([gt_mask.shape[1], gt_mask.shape[2], 3])
                gt_fin[gt_mask[0] > gt_mask[1]] = np.array([0, 255, 0])
                gt_fin[gt_mask[0] <= gt_mask[1]] = np.array([255, 0, 0])
                gt_fin[np.sum(gt_mask, axis=0) == 0] = np.array([0, 0, 0])
                
                # bg_pixels = np.sum(gt_mask, axis=0) == 0
                # gt_fin = np.argmax(gt_mask, axis = 0)
                # gt_fin[bg_pixels] = -1
                # gt_fin += 1

                np.save(os.path.join(args.viz_output_path, wandb_run_name, episodes[i], type,  "gt_" + type + "_" + str(j) + "_" + str(i)) + ".npy", gt_mask)  
                # np.save(os.path.join(args.viz_output_path, wandb_run_name, episodes[i], type, "pr_" + type + "_" + str(j) + "_" + str(i)) + ".npy", pr_mask) 
                np.save(os.path.join(args.viz_output_path, wandb_run_name, episodes[i], type, "pr_" + type + "_" + str(j) + "_" + str(i)) + ".npy", mask) 
                
                gt_im = Image.fromarray(np.uint8(gt_fin))
                # gt_im.save(args.viz_output_path + "/gt_train_mask_" + str(j) + "_" + str(i) + ".png")
                # pr_im = Image.fromarray(np.uint8(pr_fin))
                print(mask.shape)
                # pr_im = Image.fromarray(np.uint8(mask))
                # pr_im.save(args.viz_output_path + "/pr_train_mask_" + str(j) + "_" + str(i) + ".png")
                
                viz_inputs_with_gaze_overlaid([im0, im1, im2], rgb_image, gt_im, mask[0], os.path.join(args.viz_output_path, wandb_run_name, episodes[i], type, type + "_" + str(j) + "_" + str(i)))


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
    # torch.manual_seed(args.random_seed)
    # np.random.seed(args.random_seed)
    # np.random.shuffle(episode_list)
    train_episodes = episode_list[:-num_val_episodes]
    # train_episodes = ["cbdr10-23", "cbdr2-11", "cbdr2-32", "cbdr2-35", "cbdr2-41"]
    # train_episodes = ["cbdr10-36", "cbdr10-53"]
    # train_episodes = ["cbdr10-36"]
    print("Train routes:", train_episodes)
    val_episodes = episode_list[-num_val_episodes:]
    # val_episodes = ["brady-71", "abd-32", "brady-32", "abd-21"]
    # val_episodes = ["cbdr10-36", "cbdr10-53"]
    # val_episodes = ["cbdr10-36"]
    print("Val routes:", val_episodes)

    wandb_run_name = "%s_m%s_rgb%s_seg%d_sh%.1f@%.1f_g%.1f_gf%s_sample_%s" % (ENCODER, args.middle_andsides, args.use_rgb,
            args.instseg_channels, args.secs_of_history, 
            args.history_sample_rate, args.gaze_gaussian_sigma, args.gaze_fade, args.sample_clicks) + args.run_name

    # import ipdb; ipdb.set_trace()
    # best_model = torch.load("./pretrained_models/best_model_" + wandb_run_name + '.pth')
    best_model = None

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

    # train_data = []
    # for ep in train_episodes:
    #     dataset = SituationalAwarenessDataset(args.raw_data, args.sensor_config_file, ep, args)
    #     train_data.append(dataset)
    # # dataloader shuffle might just be shuffling the episode level SituationalAwarenessDataset.
    # train_dataset = torch.utils.data.ConcatDataset(train_data)

    valid_data = []
    for ep in val_episodes:
        dataset = SituationalAwarenessDataset(args.raw_data, args.sensor_config_file, ep, args)
        valid_data.append(dataset)
    valid_dataset = torch.utils.data.ConcatDataset(valid_data)

    if os.path.exists(args.viz_output_path) == False:
        os.makedirs(args.viz_output_path)
    
    if os.path.exists(os.path.join(args.viz_output_path, wandb_run_name)) == False:
        os.makedirs(os.path.join(args.viz_output_path, wandb_run_name))
     
    # # ## Visualize predictions

    # save_outputs(train_data, train_episodes, best_model, wandb_run_name, DEVICE, "train")
    save_outputs(valid_data, val_episodes, best_model, wandb_run_name, DEVICE, "valid")



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
    args.add_argument("--sample-clicks", choices=['post_click', 'pre_excl', 'both', ''], 
                    default='', help="Empty string -> sample everything")
    args.add_argument("--ignore-oldclicks", action='store_true')
    


    # training params
    args.add_argument("--device", type=str, default='cuda')
    args.add_argument("--random-seed", type=int, default=999)
    args.add_argument("--num-workers", type=int, default=12)
    args.add_argument("--batch-size", type=int, default=16)
    args.add_argument("--num-val-episodes", type=int, default=5)
    args.add_argument("--num-epochs", type=int, default=40)
    args.add_argument("--lr", type=float, default=0.0001)
    args.add_argument("--wandb", action='store_true')
    args.add_argument("--viz-output-path", type=str, default="./visualizations")
    
    args.add_argument("--run-name", type=str, default="")    
    args = args.parse_args()

    main(args)