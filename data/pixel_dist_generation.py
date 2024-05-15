import carla
from pathlib import Path
from PIL import Image
import configparser
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse 
from dataset_full import read_corrected_csv, get_clicked_frame_dict

# some constants
rgb_frame_delay = 3


def calc_pixel_dist_img(inst_seg_img, visible_ids, offset, awareness_labels):
    num_channels = 2 # aware, not aware
    pixel_dist = np.zeros(num_channels, np.int32)
    class_dist = np.zeros(num_channels, np.int32)
    object_wise_pixel_dict = {}
    visible_ids = visible_ids[visible_ids != 0]

    raw_img = np.array(inst_seg_img)
    b = raw_img[:, :, 2]
    g = raw_img[:, :, 1]
    # Calculate the sum of b*256 + g
    sum_bg = (b * 256) + g
    
    for id_idx, id in enumerate(visible_ids):
        run_id = id - offset
        obj_pixels  = sum_bg == int(run_id)
        num_obj_pixels = np.sum(obj_pixels)
        if num_obj_pixels == 0:
            continue
        pixel_dist[awareness_labels[id_idx]] += num_obj_pixels
        class_dist[awareness_labels[id_idx]] += 1
        object_wise_pixel_dict[id] = (num_obj_pixels, awareness_labels[id_idx])        

    return pixel_dist, class_dist, object_wise_pixel_dict

def calc_pixel_dist_dir(episode_dirname: Path):
    
    raw_data_dir = episode_dirname.parent
    images_dir = episode_dirname / "images"

    aw_df_file_name = episode_dirname / 'rec_parse-awdata.json'
    awareness_df = pd.read_json(aw_df_file_name)

    corrected_labels_df_filename = episode_dirname / 'corrected-awlabels.csv'
    corrected_labels_df = read_corrected_csv(corrected_labels_df_filename)
    clicked_frame_dict = get_clicked_frame_dict(corrected_labels_df)
    
    with open("%s/offset.txt" % images_dir, 'r') as file:
        offset = int(file.read()) 

    
    index_mapping = {}
    class_counts_dict = {}        
    obj_aware_dict = {}
    obj_unaware_dict = {}

    idx = 0        
    instance_seg_dir = episode_dirname /  "images" / "instance_segmentation_output" 
    label_correction_end_buffer = -50
    instance_seg_imgs = sorted(os.listdir(instance_seg_dir), key = lambda x: int(x.split('.')[0]))[:label_correction_end_buffer]
        
    obj_class_distribution = np.array([0,0], dtype=np.int64)
    pixel_class_distribution = np.array([0,0], dtype=np.int64)
    object_wise_pixel_dict_ep = {}
    if args.middle_andsides:
        object_wise_pixel_dict_ep_l = {}
        object_wise_pixel_dict_ep_r = {}
    skip_ctr = 0

    print(episode_dirname, len(instance_seg_imgs))
    # start from above index here and create index mapping to frames
    for i in range(rgb_frame_delay+1, len(instance_seg_imgs)):
        file_name = instance_seg_imgs[i] # '000001.png' for example where 1 is the frame num
        frame_num = int(file_name.split('.')[0])        

        # check that target (label mask) is not empty
        # corrected df idcs 1 removed from aw df
        cur_corrected_row = corrected_labels_df.iloc[frame_num - rgb_frame_delay - 1]            
        actor_IDs = cur_corrected_row['visible_is']
        valid_vis = actor_IDs != 0
        num_objs_vis = np.sum(valid_vis)
        if num_objs_vis == 0:
            skip_ctr += 1
            continue
    
        # calculate aware/unaware distribution
        valid_actors = actor_IDs[valid_vis]
        vis_total = cur_corrected_row['visible_total']
        vis_idcs_forlabel = np.in1d(vis_total, valid_actors)
        cur_awlabels = cur_corrected_row['awareness_label'][vis_idcs_forlabel]
        num_vis_aware = np.count_nonzero(cur_awlabels)
        num_vis_unaware = cur_awlabels.size - num_vis_aware
        obj_class_distribution += [num_vis_unaware, num_vis_aware]
    
        if num_vis_unaware > 0:
            obj_unaware_dict[idx] = num_vis_unaware 
        else:
            obj_aware_dict[idx] = num_vis_aware
        class_counts_dict[idx] = [num_vis_unaware, num_vis_aware]

        # add valid frame to index mapping
        index_mapping[idx] = frame_num
        idx += 1

        # calculate the pixels for each visible object
        visible_total = corrected_labels_df['visible_total'][frame_num-rgb_frame_delay - 1] # -1 because corrected df is one frame shifted from awareness df
        awareness_label = corrected_labels_df['awareness_label'][frame_num-rgb_frame_delay - 1] # -1 because corrected df is one frame shifted from awareness df

        instance_seg_image = Image.open(images_dir / 'instance_segmentation_output' / ('%.6d.png' % frame_num)).convert('RGB')
        pixel_dist, class_dist, object_wise_pixel_dict = calc_pixel_dist_img(instance_seg_image, visible_total, offset, awareness_label)
        pixel_class_distribution += pixel_dist
        object_wise_pixel_dict_ep[frame_num] = object_wise_pixel_dict

        # TODO : save the objwise pixel dist to a npy somewhere so we can read later

        if args.middle_andsides:
            instance_seg_left_image = Image.open(images_dir / 'instance_segmentation_output_left' / ('%.6d.png' % frame_num)).convert('RGB')
            pixel_dist, class_dist, object_wise_pixel_dict = calc_pixel_dist_img(instance_seg_left_image, visible_total, offset, awareness_label)
            object_wise_pixel_dict_ep_l[frame_num] = object_wise_pixel_dict

            instance_seg_right_image = Image.open(images_dir / 'instance_segmentation_output_right' / ('%.6d.png' % frame_num)).convert('RGB')
            pixel_dist, class_dist, object_wise_pixel_dict = calc_pixel_dist_img(instance_seg_left_image, visible_total, offset, awareness_label)
            object_wise_pixel_dict_ep_r[frame_num] = object_wise_pixel_dict
        
        print(pixel_class_distribution, end="\r", flush=True)
    
    print("\nSkipped %d frames" % skip_ctr)
    np.save(images_dir / 'objwise_pixel_dist_middle.npy', object_wise_pixel_dict_ep)

    if args.middle_andsides:
        np.save(images_dir / 'objwise_pixel_dist_left.npy', object_wise_pixel_dict_ep_l)
        np.save(images_dir / 'objwise_pixel_dist_right.npy', object_wise_pixel_dict_ep_r)       

    return


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # model params
    args.add_argument("--sensor-config-file", type=str, default='sensor_config.ini')
    args.add_argument("--raw-data", type=str, default='/media/storage/raw_data_corrected')
    args.add_argument("--middle-andsides", action='store_false')

    args = args.parse_args()

    raw_data = Path(args.raw_data)
    for episode_dir in raw_data.iterdir():
        calc_pixel_dist_dir(episode_dir)