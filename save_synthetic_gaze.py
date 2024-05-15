import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from multiprocessing import Pool

def get_synthetic_gaze_point(inst_img, visible_ids, awareness_labels, offset, gaze_points_per_frame=25):
        width, height = inst_img.size
        #mask = np.zeros((width, height))
        raw_img = np.array(inst_img)
        b = raw_img[:, :, 2]
        g = raw_img[:, :, 1]
        # Calculate the sum of b*256 + g
        sum_bg = (b * 256) + g
        
        gaze_points = []
        aware_visible_ids = visible_ids[awareness_labels == 1]
        np.random.shuffle(aware_visible_ids)
        sum_inds = 0
        for id_idx, id in enumerate(aware_visible_ids):
            run_id = id - offset
            inds = np.argwhere(sum_bg == int(run_id))
            sum_inds += len(inds)
        for id_idx, id in enumerate(aware_visible_ids):
            run_id = id - offset
            inds = np.argwhere(sum_bg == int(run_id))
            if len(inds) == 0:
                continue
            gaze_points.append(inds[np.random.choice(np.arange(len(inds)), int((len(inds)/sum_inds)*gaze_points_per_frame), replace=True)])
        # if no aware object in the scene, put gaze points on background objects
        bg = np.ones_like(sum_bg)
        for id_idx, id in enumerate(visible_ids):
            run_id = id - offset
            bg = bg*(sum_bg != int(run_id)) 
        bg_inds = np.argwhere(bg)
        gaze_points.append(bg_inds[np.random.choice(np.arange(len(bg_inds)), 15, replace=False)])
        return np.array([j[::-1] for i in gaze_points for j in i])

def read_corrected_csv(path):
    corrected_labels_df = pd.read_csv(path)
    # Extract columns into numpy arrays
    frame_no = corrected_labels_df['frame_no'].to_numpy()
    visible_total = corrected_labels_df['visible_total'].apply(eval).apply(np.array).to_numpy()
    visible_is = corrected_labels_df['visible_is'].apply(eval).apply(np.array).to_numpy()
    awareness_label = corrected_labels_df['awareness_label'].apply(eval).apply(np.array).to_numpy()
    
    
    data = {'frame_no': frame_no,
    'visible_total': visible_total,
    'awareness_label': awareness_label,
    'visible_is': visible_is}

    return pd.DataFrame(data)


def process_episode(episode):
    episode_dir = os.path.join(raw_data_dir, episode)
    images_dir = Path(os.path.join(episode_dir, "images"))
    instance_seg_dir = Path(os.path.join(images_dir, "instance_segmentation_output"))
    instance_seg_imgs = sorted(os.listdir(instance_seg_dir), key=lambda x: int(x.split('.')[0]))
    corrected_labels_df_filename = os.path.join(episode_dir, 'corrected-awlabels.csv')
    corrected_labels_df = read_corrected_csv(corrected_labels_df_filename)
    synthetic_gaze_points_dict = {}

    for i in range(4, len(instance_seg_imgs)):
        try:
            # print(episode, i, len(instance_seg_imgs))
            file_name = instance_seg_imgs[i]
            frame_num = int(file_name.split('.')[0])
            cur_corrected_row = corrected_labels_df.iloc[frame_num - rgb_frame_delay - 1]
            visible_total = cur_corrected_row['visible_total']
            awareness_label = cur_corrected_row['awareness_label']

            with open(os.path.join(images_dir, 'offset.txt'), 'r') as file:
                offset = int(file.read())

            instance_seg_image = Image.open(instance_seg_dir / ('%.6d.png' % frame_num)).convert('RGB')
            instance_seg_left_image = Image.open(images_dir / 'instance_segmentation_output_left' / ('%.6d.png' % frame_num)).convert('RGB')
            instance_seg_right_image = Image.open(images_dir / 'instance_segmentation_output_right' / ('%.6d.png' % frame_num)).convert('RGB')

            synthetic_gaze_points_dict[frame_num] = {'mid': [], 'left': [], 'right': []}
            synthetic_gaze_points_dict[frame_num]['mid'] = get_synthetic_gaze_point(instance_seg_image, visible_total, awareness_label, offset, 25)
            synthetic_gaze_points_dict[frame_num]['left'] = get_synthetic_gaze_point(instance_seg_left_image, visible_total, awareness_label, offset, 25)
            synthetic_gaze_points_dict[frame_num]['right'] = get_synthetic_gaze_point(instance_seg_right_image, visible_total, awareness_label, offset, 25)
        except Exception as e:
            print(e)
    np.save(os.path.join(episode_dir, 'synthetic_gaze_points.npy'), np.array([synthetic_gaze_points_dict]))
    print(episode)
    return episode

def check_processed_episode(episode):
    episode_dir = os.path.join(raw_data_dir, episode)
    dict = np.load(os.path.join(episode_dir, 'synthetic_gaze_points.npy'), allow_pickle=True).item()
    print(episode, len(dict.keys()))

if __name__ == '__main__':
    raw_data_dir = "/home/harpadmin/raw_data_corrected"
    rgb_frame_delay = 3
    episodes = os.listdir(raw_data_dir)

    with Pool(processes=20) as pool:
        processed_episodes = pool.map(process_episode, episodes)

    print("All episodes processed.")

# import numpy as np
# import os
# import pandas as pd
# from pathlib import Path
# from PIL import Image

# def get_synthetic_gaze_point(inst_img, visible_ids, awareness_labels, offset, gaze_points_per_frame=25):
#         width, height = inst_img.size
#         #mask = np.zeros((width, height))
#         raw_img = np.array(inst_img)
#         b = raw_img[:, :, 2]
#         g = raw_img[:, :, 1]
#         # Calculate the sum of b*256 + g
#         sum_bg = (b * 256) + g
        
#         gaze_points = []
#         aware_visible_ids = visible_ids[awareness_labels == 1]
#         np.random.shuffle(aware_visible_ids)
#         sum_inds = 0
#         for id_idx, id in enumerate(aware_visible_ids):
#             run_id = id - offset
#             inds = np.argwhere(sum_bg == int(run_id))
#             sum_inds += len(inds)
#         for id_idx, id in enumerate(aware_visible_ids):
#             run_id = id - offset
#             inds = np.argwhere(sum_bg == int(run_id))
#             if len(inds) == 0:
#                 continue
#             gaze_points.append(inds[np.random.choice(np.arange(len(inds)), int((len(inds)/sum_inds)*gaze_points_per_frame), replace=True)])
        
#         return np.array([j[::-1] for i in gaze_points for j in i])

# def read_corrected_csv(path):
#     corrected_labels_df = pd.read_csv(path)
#     # Extract columns into numpy arrays
#     frame_no = corrected_labels_df['frame_no'].to_numpy()
#     visible_total = corrected_labels_df['visible_total'].apply(eval).apply(np.array).to_numpy()
#     visible_is = corrected_labels_df['visible_is'].apply(eval).apply(np.array).to_numpy()
#     awareness_label = corrected_labels_df['awareness_label'].apply(eval).apply(np.array).to_numpy()
    
    
#     data = {'frame_no': frame_no,
#     'visible_total': visible_total,
#     'awareness_label': awareness_label,
#     'visible_is': visible_is}

#     return pd.DataFrame(data)

# raw_data_dir = "/home/harpadmin/raw_data_corrected"
# rgb_frame_delay = 3

# for episode in os.listdir(raw_data_dir):
#     print(episode)
#     images_dir = Path(os.path.join(raw_data_dir, episode, "images"))    
#     instance_seg_dir = Path(os.path.join(raw_data_dir, episode, "images", "instance_segmentation_output"))
#     instance_seg_imgs = sorted(os.listdir(instance_seg_dir), key = lambda x: int(x.split('.')[0]))
#     corrected_labels_df_filename = os.path.join(raw_data_dir, episode, 'corrected-awlabels.csv')
#     corrected_labels_df = read_corrected_csv(corrected_labels_df_filename)
#     synthetic_gaze_points_dict = {}

#     for i in range(4, len(instance_seg_imgs)):
#         file_name = instance_seg_imgs[i] # '000001.png' for example where 1 is the frame num
#         frame_num = int(file_name.split('.')[0])        

#         # check that target (label mask) is not empty
#         # corrected df idcs 1 removed from aw df
#         cur_corrected_row = corrected_labels_df.iloc[frame_num - rgb_frame_delay - 1]            
        
#         # save synthetic gaze points
        
#         visible_total = cur_corrected_row['visible_total']
#         awareness_label = cur_corrected_row['awareness_label']

#         with open("%s/offset.txt" % images_dir, 'r') as file:
#             offset = int(file.read())

#         instance_seg_image = Image.open(images_dir / 'instance_segmentation_output' / ('%.6d.png' % frame_num)).convert('RGB')
#         instance_seg_left_image = Image.open(images_dir / 'instance_segmentation_output_left' / ('%.6d.png' % frame_num)).convert('RGB')
#         instance_seg_right_image = Image.open(images_dir / 'instance_segmentation_output_right' / ('%.6d.png' % frame_num)).convert('RGB')

#         synthetic_gaze_points_dict[frame_num] = {'mid':[], 'left':[], 'right':[]}
#         synthetic_gaze_points_dict[frame_num]['mid'] = get_synthetic_gaze_point(instance_seg_image, visible_total, awareness_label, offset, 25)
#         synthetic_gaze_points_dict[frame_num]['left'] = get_synthetic_gaze_point(instance_seg_left_image, visible_total, awareness_label, offset, 25)
#         synthetic_gaze_points_dict[frame_num]['right'] = get_synthetic_gaze_point(instance_seg_right_image, visible_total, awareness_label, offset, 25)
#         print(i, len(instance_seg_imgs))
#     np.save(os.path.join(raw_data_dir, episode, 'synthetic_gaze_points.npy'), np.array([synthetic_gaze_points_dict]))
