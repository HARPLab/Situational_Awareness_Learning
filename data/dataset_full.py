import torch
import carla
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import configparser
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from io import BytesIO
import torchvision.transforms.functional as TF

def ptsWorld2Cam(focus_hit_pt, world2camMatrix, K):
    tick_focus_hitpt_homog = np.hstack((focus_hit_pt,1))    
    sensor_points = np.dot(world2camMatrix, tick_focus_hitpt_homog)
    
    # Now we must change from UE4's coordinate system to an "standard" camera coordinate system
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]])

    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, point_in_camera_coords)

    # Remember to normalize the x, y values by the 3rd value.
    points_2d /= points_2d[2]

    u_coord = points_2d[0].astype(np.int)
    v_coord = points_2d[1].astype(np.int)
    return (u_coord, v_coord)

def world2pixels(focus_hit_pt, vehicle_transform, K, sensor_config):
    '''
    takes in the dataframe row with all the information of where the world is currently 
    '''        
    vehicleP = vehicle_transform.get_matrix()
    
    # center image
    camera_loc_offset = carla.Location(x=float(sensor_config['rgb']['x']), y=float(sensor_config['rgb']['y']), z=float(sensor_config['rgb']['z']))    
    camera_rot_offset = carla.Rotation(pitch=float(sensor_config['rgb']['pitch']), yaw=float(sensor_config['rgb']['yaw']), roll=float(sensor_config['rgb']['roll']))
    cam_transform = carla.Transform(location=camera_loc_offset, rotation=camera_rot_offset)
    world2cam = np.matmul(cam_transform.get_inverse_matrix(), vehicle_transform.get_inverse_matrix())    
    
    u,v = ptsWorld2Cam(focus_hit_pt, world2cam, K)
    pts_mid = (u,v)
        
    # left image  
    camera_loc_offset = carla.Location(x=float(sensor_config['rgb_left']['x']), y=float(sensor_config['rgb_left']['y']), z=float(sensor_config['rgb_left']['z']))    
    camera_rot_offset = carla.Rotation(pitch=float(sensor_config['rgb_left']['pitch']), yaw=float(sensor_config['rgb_left']['yaw']), roll=float(sensor_config['rgb_left']['roll']))
    cam_transform = carla.Transform(location=camera_loc_offset, rotation=camera_rot_offset)    
    world2cam = np.matmul(cam_transform.get_inverse_matrix(), vehicle_transform.get_inverse_matrix())
        
    u,v = ptsWorld2Cam(focus_hit_pt, world2cam, K)
    pts_left = (u,v)
    
    # right image  
    camera_loc_offset = carla.Location(x=float(sensor_config['rgb_right']['x']), y=float(sensor_config['rgb_right']['y']), z=float(sensor_config['rgb_right']['z']))    
    camera_rot_offset = carla.Rotation(pitch=float(sensor_config['rgb_right']['pitch']), yaw=float(sensor_config['rgb_right']['yaw']), roll=float(sensor_config['rgb_right']['roll']))
    cam_transform = carla.Transform(location=camera_loc_offset, rotation=camera_rot_offset)    
    world2cam = np.matmul(cam_transform.get_inverse_matrix(), vehicle_transform.get_inverse_matrix())
        
    u,v = ptsWorld2Cam(focus_hit_pt, world2cam, K)
    pts_right = (u,v)    
    
    return pts_mid, pts_left, pts_right

def gaussian_contour_plot(rgb_image, gaze_points, sigma=10, kernel_size = 41, gaze_fade=False):
    # Create a grid of coordinates
    height, width = rgb_image.shape[:2]
    mask = torch.zeros((1, height, width), dtype=torch.uint8)

    gaze_fade_min = 10    
    N = len(gaze_points)
    gaze_fade_step = (255 - gaze_fade_min) / N
    
    if not gaze_fade:
        for center_pixel in gaze_points:
            mask[0, center_pixel[1], center_pixel[0]] = 255
    else:
        for i, center_pixel in enumerate(gaze_points):
            if center_pixel[0] < -1 or center_pixel[1] < -1:
                continue
            # fade gaze from 255 to gaze_fade_min linearly based on time
            # most recent gaze is gaze_points[0]
            mask[0, center_pixel[1], center_pixel[0]] = int(gaze_fade_min + gaze_fade_step*(N-i))

    # convolve the mask with a gaussian kernel    
    heatmap = TF.gaussian_blur(mask, kernel_size, sigma)
    if not gaze_fade:
        heatmap[heatmap > 0] = 255

    # heatmap_image = Image.fromarray(heatmap.numpy().squeeze())
    # heatmap_image.save('heatmap.png')
    return heatmap

def frame_filter(frame_num, awareness_df):
     ego_vel = awareness_df["EgoVariables_VehicleVel"][frame_num]
     visible_list = awareness_df["AwarenessData_Visible"][frame_num]
     if ego_vel == 0 and visible_list == []:
          return 0
     else:
          return 1

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

class SituationalAwarenessDataset(Dataset):
    #def __init__(self, images_dir, awareness_df, sensor_config, secs_of_history = 5, sample_rate = 4.0, gaussian_sigma = 10.0):
    def __init__(self, raw_data, sensor_config, episode, args):
        self.raw_data_dir = Path(raw_data)
        self.episode = episode
        self.rgb_frame_delay = 3
        self.use_rgb = args.use_rgb
        self.middle_andsides = args.middle_andsides
        self.instseg_channels = args.instseg_channels
        self.images_dir = Path(os.path.join(self.raw_data_dir, self.episode, "images"))
        aw_df_file_name = os.path.join(self.raw_data_dir, self.episode, 'rec_parse-awdata.json')
        self.awareness_df = pd.read_json(aw_df_file_name, orient='index')
        self.args = args

        secs_of_history = args.secs_of_history
        sample_rate = args.history_sample_rate
        gaussian_sigma = args.gaze_gaussian_sigma
        
        corrected_labels_df_filename = os.path.join(self.raw_data_dir, self.episode, 'corrected-awlabels.csv')
        self.corrected_labels_df = read_corrected_csv(corrected_labels_df_filename)

        self.clicked_frame_dict = get_clicked_frame_dict(self.corrected_labels_df)
        
        if self.args.sample_clicks:
            # max time allowed since last click for valid sample
            self.sample_clicks_min_time = 10 #self.args.secs_of_history                
            # skip samples just before click (to account for reaction time)
            self.pre_clicks_excl_time = 1

            # precompute userinput frame nos
            user_input_arr = np.array(self.awareness_df.AwarenessData_UserInput)
            user_input_locs = np.nonzero(user_input_arr)[0] # returning 
            user_input_diffs = np.diff(user_input_locs)
            user_input_diffs = np.insert(user_input_diffs, 0, user_input_locs[0])
            self.user_input_locs = user_input_locs[user_input_diffs > 7] # 6 clicks occur together for one press
            self.prev_click_idx = 0
            self.prev_click_time = self.awareness_df.iloc[self.user_input_locs[self.prev_click_idx]]['TimeElapsed']
            self.next_click_idx = 0
            self.next_click_time = self.awareness_df.iloc[self.user_input_locs[self.next_click_idx]]['TimeElapsed']

        index_mapping = {}
        idx = 0
            
        instance_seg_dir = Path(os.path.join(self.raw_data_dir, self.episode, "images", "instance_segmentation_output"))
        label_correction_end_buffer = -50
        instance_seg_imgs = sorted(os.listdir(instance_seg_dir), key = lambda x: int(x.split('.')[0]))[:label_correction_end_buffer]
        
        first_valid_idx = self.awareness_df[self.awareness_df["TimeElapsed"] > secs_of_history].index[0]
        class_distribution = np.array([0,0])
        skip_ctr = 0

        # start from above index here
        for i in range(first_valid_idx, len(instance_seg_imgs)):
            file_name = instance_seg_imgs[i] # '000001.png' for example where 1 is the frame num
            frame_num = int(file_name.split('.')[0])        

            # check that target (label mask) is not empty
            # corrected df idcs 1 removed from aw df
            cur_corrected_row = self.corrected_labels_df.iloc[frame_num - self.rgb_frame_delay - 1]            
            actor_IDs = cur_corrected_row['visible_is']
            valid_vis = actor_IDs != 0
            num_objs_vis = np.sum(valid_vis)
            if num_objs_vis == 0:
                skip_ctr += 1
                continue

            # if we want to sample only around clicks, 
            if self.args.sample_clicks:                
                # find previous and next click
                # idx = np.searchsorted(self.user_input_locs, frame_num - self.rgb_frame_delay)
                # returned index i satisfies a[i-1] < v <= a[i]

                cur_click_time = self.awareness_df.iloc[frame_num - self.rgb_frame_delay]['TimeElapsed']
                
                if cur_click_time > self.next_click_time:
                    self.prev_click_idx = self.next_click_idx
                    self.prev_click_time = self.next_click_time
                    
                    if self.next_click_idx == len(self.user_input_locs)-1:
                        pass
                    else:
                        self.next_click_idx += 1
                        self.next_click_time = self.awareness_df.iloc[self.user_input_locs[self.next_click_idx]]['TimeElapsed']                    

                if self.args.sample_clicks == 'both':
                    if self.next_click_time - cur_click_time < self.pre_clicks_excl_time:
                        if cur_click_time - self.prev_click_time < self.sample_clicks_min_time: # okay even if this is negative
                            # sample
                            pass
                        else:
                            skip_ctr +=1
                            continue
                    elif cur_click_time - self.prev_click_time >= self.sample_clicks_min_time:
                        # sample
                        skip_ctr +=1
                        continue
                elif self.args.sample_clicks == 'post_click':
                    if cur_click_time - self.prev_click_time < self.sample_clicks_min_time: # okay even if this is negative
                        # sample
                        pass
                    else:
                        skip_ctr +=1
                        continue
                elif self.args.sample_clicks == 'pre_excl':
                    if self.next_click_time - cur_click_time < self.pre_clicks_excl_time:                
                        skip_ctr +=1
                        continue
                    else:
                        # sample
                        pass

            # calculate aware/unaware distribution
            valid_actors = actor_IDs[valid_vis]
            vis_total = cur_corrected_row['visible_total']
            vis_idcs_forlabel = np.in1d(vis_total, valid_actors)
            cur_awlabels = cur_corrected_row['awareness_label'][vis_idcs_forlabel]            
            num_vis_aware = np.count_nonzero(cur_awlabels)
            num_vis_unaware = cur_awlabels.size - num_vis_aware
            class_distribution += [num_vis_unaware, num_vis_aware]

            # add valid frame to index mapping
            index_mapping[idx] = frame_num

            idx += 1
        
        self.index_mapping = index_mapping
        #print(self.index_mapping)
        
        #self.images_dir = Path(images_dir)
        self.sensor_config = configparser.ConfigParser()
        self.sensor_config.read(sensor_config)
        #self.awareness_df = awareness_df
        self.secs_of_history = secs_of_history 
        self.sample_rate = sample_rate
        self.gaussian_sigma = gaussian_sigma
        print("Class distribution for episode %s: %s" % (self.episode, class_distribution))
        print("Number of frames skipped: %d" % skip_ctr)
    
    def __getitem__(self, idx):
        # Return rgb_img, instance_segmentation_img, gaze_heatmap 
        # (read in raw gaze and construct heatmap in get_item itself), label mask image
        # idx = max(self.index_mapping.keys())
        frame_num = self.index_mapping[idx]
        # print(frame_num)

        # init data paths
        if self.use_rgb:        
            rgb_image = Image.open(self.images_dir / 'rgb_output' / ('%.6d.png' % frame_num)).convert('RGB')
            if not self.middle_andsides:
                rgb_left_image = Image.open(self.images_dir / 'rgb_output_left' / ('%.6d.png' % frame_num)).convert('RGB')
                rgb_right_image = Image.open(self.images_dir / 'rgb_output_right' / ('%.6d.png' % frame_num)).convert('RGB')
             

        instance_seg_image = Image.open(self.images_dir / 'instance_segmentation_output' / ('%.6d.png' % frame_num)).convert('RGB')
        if not self.middle_andsides:
            instance_seg_left_image = Image.open(self.images_dir / 'instance_segmentation_output_left' / ('%.6d.png' % frame_num)).convert('RGB')
            instance_seg_right_image = Image.open(self.images_dir / 'instance_segmentation_output_right' / ('%.6d.png' % frame_num)).convert('RGB')

        # Label mask generation
        # id_list = self.awareness_df["AwarenessData_Visible"][frame_num-self.rgb_frame_delay]
        # aw_visible = self.awareness_df["AwarenessData_Visible"][frame_num-self.rgb_frame_delay]
        # user_input = self.awareness_df["AwarenessData_UserInput"][frame_num-self.rgb_frame_delay]
        # aw_answer = self.awareness_df["AwarenessData_Answer"][frame_num-self.rgb_frame_delay]

        visible_total = self.corrected_labels_df['visible_total'][frame_num-self.rgb_frame_delay - 1] # -1 because corrected df is one frame shifted from awareness df
        awareness_label = self.corrected_labels_df['awareness_label'][frame_num-self.rgb_frame_delay - 1] # -1 because corrected df is one frame shifted from awareness df

        with open("%s/offset.txt" % self.images_dir, 'r') as file:
            offset = int(file.read()) 
        
        # full_label_mask = self.get_full_label_mask(instance_seg_image, id_list, offset, aw_visible, aw_answer, user_input)        
        full_label_mask = self.get_corrected_label_mask(instance_seg_image, visible_total, awareness_label, offset=offset)
        if not self.middle_andsides:
            full_label_mask_left = self.get_corrected_label_mask(instance_seg_left_image, visible_total, awareness_label, offset=offset)
            full_label_mask_right = self.get_corrected_label_mask(instance_seg_right_image, visible_total, awareness_label, offset=offset)
        # label_mask_image = Image.fromarray(full_label_mask)

        if self.args.ignore_oldclicks:
            ignore_mask = self.get_ignore_mask(instance_seg_image, visible_total, awareness_label, offset=offset)
            if not self.middle_andsides:
                ignore_mask_left = self.get_ignore_mask(instance_seg_left_image, visible_total, awareness_label, offset=offset)
                ignore_mask_right = self.get_ignore_mask(instance_seg_right_image, visible_total, awareness_label, offset=offset)
        else:
            ignore_mask = np.ones_like(full_label_mask)
            if not self.middle_andsides:
                ignore_mask_left = np.ones_like(full_label_mask_left)
                ignore_mask_right = np.ones_like(full_label_mask_right)
        


        # Construct gaze heatmap
        raw_gaze_mid= []
        if not self.middle_andsides:
            raw_gaze_left = []
            raw_gaze_right = []
        
        current_time = self.awareness_df["TimeElapsed"][frame_num - self.rgb_frame_delay]
        end_time = current_time - self.secs_of_history
        history_frames = []
        history_frame_times = []
        i = 0
        frame = frame_num - i - 1
        while frame > 0 and self.awareness_df["TimeElapsed"][frame - self.rgb_frame_delay] > end_time:
            history_frames.append(frame)
            history_frame_times.append(self.awareness_df["TimeElapsed"][frame - self.rgb_frame_delay])
            frame = frame_num - i - 1
            i+=1
        
        assert self.secs_of_history*self.sample_rate == int(self.secs_of_history*self.sample_rate), "sample_rate x secs_of_history must be an integer"
        total_history_frames = int(self.secs_of_history*self.sample_rate)
        frame_time = 1/self.sample_rate
        step = 0

        frame = frame_num - 1
        raw_gaze_mid, raw_gaze_left, raw_gaze_right = np.zeros((total_history_frames+1, 2), dtype=np.int32)-1,  \
                                                        np.zeros((total_history_frames+1, 2), dtype=np.int32)-1, \
                                                        np.zeros((total_history_frames+1, 2), dtype=np.int32)-1
        while frame - self.rgb_frame_delay > 0 and step != total_history_frames:

            target_time = self.awareness_df["TimeElapsed"][frame - self.rgb_frame_delay] - frame_time*step
            closest_frame_indices_ranking = np.argsort(abs(np.array(history_frame_times)-target_time))
            closest_frame_ranking = [history_frames[i] for i in closest_frame_indices_ranking]
            
            closest_frame = 0
            for f in range(len(closest_frame_ranking)):
                if os.path.exists(self.images_dir / 'instance_segmentation_output' / \
                                  ('%.6d.png' % (closest_frame_ranking[f] + self.rgb_frame_delay))):
                    closest_frame = closest_frame_ranking[f]
                    break
            step += 1

            focus_hit_pt_i = np.asarray(self.awareness_df["FocusInfo_HitPoint"][closest_frame])
            loc = self.awareness_df["EgoVariables_VehicleLoc"][closest_frame]
            loc = np.asarray([loc[0], loc[1], loc[2]])
            rot = self.awareness_df["EgoVariables_VehicleRot"][closest_frame]
            rot = np.asarray([rot[0], rot[1], rot[2]])
            
            pts_mid, pts_left, pts_right = self.get_gaze_point(focus_hit_pt_i, loc, rot)
            

            # don't append gaze if its outside the image            
            if pts_mid[0] >= 0 and pts_mid[0] < instance_seg_image.width and \
                pts_mid[1] >= 0 and pts_mid[1] < instance_seg_image.height:
                raw_gaze_mid[step,:] = pts_mid
            if not self.middle_andsides:
                if pts_left[0] >= 0 and pts_left[0] < instance_seg_left_image.width and \
                    pts_left[1] >= 0 and pts_left[1] < instance_seg_left_image.height:
                    raw_gaze_left[step,:] = pts_left
                if pts_right[0] >= 0 and pts_right[0] < instance_seg_right_image.width and \
                    pts_right[1] >= 0 and pts_right[1] < instance_seg_right_image.height:
                    raw_gaze_right[step,:] = pts_right

        gaze_heatmap = gaussian_contour_plot(np.array(instance_seg_image), raw_gaze_mid, sigma=self.gaussian_sigma, gaze_fade=self.args.gaze_fade)
        if not self.middle_andsides:
            gaze_heatmap_left = gaussian_contour_plot(np.array(instance_seg_left_image), 
                                                      raw_gaze_left, sigma=self.gaussian_sigma, gaze_fade=self.args.gaze_fade)
            gaze_heatmap_right = gaussian_contour_plot(np.array(instance_seg_right_image), 
                                                       raw_gaze_right, sigma=self.gaussian_sigma, gaze_fade=self.args.gaze_fade)
        
        # Convert all images to tensors
        if self.use_rgb:
            rgb_image = transforms.functional.to_tensor(rgb_image)
            if not self.middle_andsides:
                rgb_left_image = transforms.functional.to_tensor(rgb_left_image)
                rgb_right_image = transforms.functional.to_tensor(rgb_right_image)
        instance_seg_image = transforms.functional.to_tensor(instance_seg_image)[:self.instseg_channels, :, : ] # only read the r and g channel
        if not self.middle_andsides:
            instance_seg_left_image = transforms.functional.to_tensor(instance_seg_left_image)[:self.instseg_channels, :, : ]
            instance_seg_right_image = transforms.functional.to_tensor(instance_seg_right_image)[:self.instseg_channels, :, : ]
        
        label_mask_image = transforms.functional.to_tensor(full_label_mask)
        if not self.middle_andsides:
            label_mask_image_left = transforms.functional.to_tensor(full_label_mask_left)
            label_mask_image_right = transforms.functional.to_tensor(full_label_mask_right)

        ignore_mask = transforms.functional.to_tensor(ignore_mask)
        if not self.middle_andsides:
            ignore_mask_left = transforms.functional.to_tensor(ignore_mask_left)
            ignore_mask_right = transforms.functional.to_tensor(ignore_mask_right)

        if self.use_rgb:
            if not self.middle_andsides:
                input_images = (rgb_image, instance_seg_image, gaze_heatmap, 
                                rgb_left_image, instance_seg_left_image, gaze_heatmap_left, 
                                rgb_right_image, instance_seg_right_image, gaze_heatmap_right)
            else:
                input_images = (rgb_image, instance_seg_image, gaze_heatmap)

        else:
            if not self.middle_andsides:
                input_images = (instance_seg_image, gaze_heatmap, 
                                instance_seg_left_image, gaze_heatmap_left, 
                                instance_seg_right_image, gaze_heatmap_right)
            else:
                input_images = (instance_seg_image, gaze_heatmap)
        final_input_image = torch.cat(input_images)
        if not self.middle_andsides:
            final_label_mask_image = torch.cat((label_mask_image, label_mask_image_left, label_mask_image_right))
        else:
            final_label_mask_image = label_mask_image
        
        if not self.middle_andsides:
            final_ignore_mask = torch.cat((ignore_mask, ignore_mask_left, ignore_mask_right))
        else:
            final_ignore_mask = ignore_mask

        padded_tensor = torch.nn.functional.pad(final_input_image, (0, 0, 4, 4), mode='constant', value=0)
        padded_label_mask_image_tensor = torch.nn.functional.pad(final_label_mask_image, (0, 0, 4, 4), mode='constant', value=0)
        padded_final_ignore_mask = torch.nn.functional.pad(final_ignore_mask, (0, 0, 4, 4), mode='constant', value=0)

        return padded_tensor, padded_label_mask_image_tensor, padded_final_ignore_mask
    
    def get_ignore_mask(self, inst_img, visible_ids, awareness_labels, offset, frame_num):
        width, height = inst_img.size
        num_channels = 2 # aware, not aware
        mask = np.ones((height, width, num_channels), dtype=np.uint8)
        raw_img = np.array(inst_img)
        b = raw_img[:, :, 2]
        g = raw_img[:, :, 1]
        # Calculate the sum of b*256 + g
        sum_bg = (b * 256) + g
        
        for id_idx, id in enumerate(visible_ids):
            run_id = id - offset
            if awareness_labels[id_idx] == 1:
                clicked_frame = self.clicked_frame_dict[run_id]
                if frame_num - clicked_frame > self.click_recency_threshold:
                # Create a mask where sum_bg is equal to target_value
                    mask[sum_bg == int(run_id), 0] = 0
                    mask[sum_bg == int(run_id), 1] = 0
        
        return mask

    def get_corrected_label_mask(self, inst_img, visible_ids, awareness_labels, offset):
        width, height = inst_img.size
        num_channels = 2 # aware, not aware
        mask = np.zeros((height, width, num_channels), dtype=np.uint8)
        #mask = np.zeros((width, height))
        raw_img = np.array(inst_img)
        b = raw_img[:, :, 2]
        g = raw_img[:, :, 1]
        # Calculate the sum of b*256 + g
        sum_bg = (b * 256) + g
        
        for id_idx, id in enumerate(visible_ids):
            run_id = id - offset
            if awareness_labels[id_idx] == 1:
                # Create a mask where sum_bg is equal to target_value
                mask[sum_bg == int(run_id), 0] = 255
            else:
                mask[sum_bg == int(run_id), 1] = 255
        
        return mask
    
    def get_full_label_mask(self, inst_img, id_list, offset, aw_visible, aw_answer, user_input, type_bit=16):
        width, height = inst_img.size
        mask = np.zeros((height, width), dtype=np.uint8)
        #mask = np.zeros((width, height))
        raw_img = np.array(inst_img)
        b = raw_img[:, :, 2]
        g = raw_img[:, :, 1]
        # Calculate the sum of b*256 + g
        sum_bg = (b * 256) + g
        
        for id in id_list:
            run_id = id - offset
            id_idx = aw_visible.index(id)
            label = False
            if (user_input & type_bit == aw_answer[id_idx] & type_bit) and (user_input & aw_answer[id_idx]):
                label = True
            if label == True:
                # Create a mask where sum_bg is equal to target_value
                mask[sum_bg==run_id] = 100
            else:
                mask[sum_bg==run_id] = 200
        
        return mask

    def get_gaze_point(self, focus_hit_pt, loc, rot):
        FOV = int(self.sensor_config['rgb']['fov'])
        w = int(self.sensor_config['rgb']['width'])
        h = int(self.sensor_config['rgb']['height'])
        F = w / (2 * np.tan(FOV * np.pi / 360))

        cam_info = {
            'w' : w,
            'h' : h,
            'fy' : F,
            'fx' : 1.0 * F,
        }

        K = np.array([
        [cam_info['fx'], 0, cam_info['w']/2],
        [0, cam_info['fy'], cam_info['h']/2],
        [0, 0, 1]])
        
        focus_hit_pt_scaled = np.array(focus_hit_pt.squeeze())/100 # conversion cm to m 
        vehicle_loc = carla.Location(*(loc.squeeze()))/100
        vehicle_rot = carla.Rotation(*(rot.squeeze()))
        vehicle_transform = carla.Transform(location=vehicle_loc, rotation=vehicle_rot)

        return world2pixels(focus_hit_pt_scaled, vehicle_transform, K, self.sensor_config)


    def __len__(self):        
        return len(self.index_mapping)


        