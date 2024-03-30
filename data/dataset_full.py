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

def gaussian_contour_plot(rgb_image, gaze_points, sigma=10.0, contour_levels=3):
    # Create a grid of coordinates
    height, width = rgb_image.shape[:2]
    y, x = np.mgrid[0:height, 0:width]

    composite_gaussian = np.zeros((height, width), dtype=float)

    # Combine Gaussians centered at each point
    for center_pixel in gaze_points:
        mean = center_pixel
        covariance_matrix = np.eye(2) * (sigma**2)
        gaussian_distribution = multivariate_normal(mean=mean, cov=covariance_matrix)

        positions = np.column_stack((x.ravel(), y.ravel()))
        values = gaussian_distribution.pdf(positions)
        gaussian_image = values.reshape(height, width)

        composite_gaussian += gaussian_image
    buffer = BytesIO()
    # Plot the original image and overlay the composite Gaussian contour plot
    # px = 1/plt.rcParams['figure.dpi']
    # plt.figure(figsize=(600*px,800*px))
    plt.figure(figsize=(width / 100, height / 100))
    plt.axis('off')
    plt.imshow(rgb_image, cmap='gray')
    plt.contourf(x, y, composite_gaussian, levels=contour_levels, cmap='binary_r', alpha=0.7)
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    heatmap_image = Image.open(buffer).convert('L')
    
    return heatmap_image


def frame_filter(frame_num, awareness_df):
     ego_vel = awareness_df["EgoVariables_VehicleVel"][frame_num-1]
     visible_list = awareness_df["AwarenessData_Visible"][frame_num-1]
     if ego_vel == 0 and visible_list == []:
          return 0
     else:
          return 1


class SituationalAwarenessDataset(Dataset):
    #def __init__(self, images_dir, awareness_df, sensor_config, secs_of_history = 5, sample_rate = 4.0, gaussian_sigma = 10.0):
    def __init__(self, raw_data, sensor_config, episode, secs_of_history = 5, sample_rate = 4.0, gaussian_sigma = 10.0):
        self.raw_data_dir = Path(raw_data)
        self.episode = episode
        
        self.images_dir = Path(os.path.join(self.raw_data_dir, self.episode, "images"))
        aw_df_file_name = os.path.join(self.raw_data_dir, self.episode, 'rec_parse-awdata.json')
        self.awareness_df = pd.read_json(aw_df_file_name, orient='index')
        index_mapping = {}
        idx = 0
        # if self.episode == "cbdr10-23": #REMOVE THIS LATER
        #     images_dirs.append(Path(os.path.join(self.raw_data_dir, e, "images")))
        #     aw_df_file_name = os.path.join(self.raw_data_dir, e, 'rec_parse-awdata.json')
        #     aw_df_e = pd.read_json(aw_df_file_name, orient='index')
        #     awareness_dfs.append(aw_df_e)
            
        instance_seg_dir = Path(os.path.join(self.raw_data_dir, self.episode, "images", "instance_segmentation_output"))
        instance_seg_imgs = os.listdir(instance_seg_dir)
        for i in range(len(instance_seg_imgs)):
            file_name = instance_seg_imgs[i] # '000001.png' for example where 1 is the frame num
            frame_num = int(file_name.split('.')[0])
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
    
    def __getitem__(self, idx):
        # Return rgb_img, instance_segmentation_img, gaze_heatmap 
        # (read in raw gaze and construct heatmap in get_item itself), label mask image
        frame_num = self.index_mapping[idx]
        print(frame_num)
        rgb_image = Image.open(self.images_dir / 'rgb_output' / ('%.6d.png' % frame_num)).convert('RGB')
        rgb_left_image = Image.open(self.images_dir / 'rgb_output_left' / ('%.6d.png' % frame_num)).convert('RGB')
        rgb_right_image = Image.open(self.images_dir / 'rgb_output_right' / ('%.6d.png' % frame_num)).convert('RGB')

        instance_seg_image = Image.open(self.images_dir / 'instance_segmentation_output' / ('%.6d.png' % frame_num)).convert('RGB')
        instance_seg_left_image = Image.open(self.images_dir / 'instance_segmentation_output_left' / ('%.6d.png' % frame_num)).convert('RGB')
        instance_seg_right_image = Image.open(self.images_dir / 'instance_segmentation_output_right' / ('%.6d.png' % frame_num)).convert('RGB')

        id_list = self.awareness_df["AwarenessData_Visible"][frame_num]
        aw_visible = self.awareness_df["AwarenessData_Visible"][frame_num]
        user_input = self.awareness_df["AwarenessData_UserInput"][frame_num]
        aw_answer = self.awareness_df["AwarenessData_Answer"][frame_num]
        with open("%s/offset.txt" % self.images_dir, 'r') as file:
            offset = int(file.read()) 
        
        full_label_mask = self.get_full_label_mask(instance_seg_image, id_list, offset, aw_visible, aw_answer, user_input)
        label_mask_image = Image.fromarray(full_label_mask)
        
        #label_mask_image = Image.open(images_dir / 'full_label_masks' / ('%.6d.png' % (frame_num)))

        # Construct gaze heatmap
        raw_gaze_mid= []
        raw_gaze_left = []
        raw_gaze_right = []
        
        current_time = self.awareness_df["TimeElapsed"][frame_num]
        end_time = current_time - self.secs_of_history
        history_frames = []
        history_frame_times = []
        i = 0
        frame = frame_num - i - 1
        while frame > 0 and self.awareness_df["TimeElapsed"][frame] > end_time:
            history_frames.append(frame)
            history_frame_times.append(self.awareness_df["TimeElapsed"][frame])
            frame = frame_num - i - 1
            i+=1
        
        total_history_frames = self.secs_of_history*self.sample_rate
        frame_time = 1/self.sample_rate
        step = 0

        frame = frame_num - 1
        while frame > 0 and step != total_history_frames:

            target_time = self.awareness_df["TimeElapsed"][frame] - frame_time*step
            #closest_frame_idx = np.argmin(abs(history_frame_times-target_time))
            closest_frame_indices_ranking = np.argsort(abs(np.array(history_frame_times)-target_time))
            #closest_frame = history_frames[closest_frame_idx]
            closest_frame_ranking = [history_frames[i] for i in closest_frame_indices_ranking]
            
            closest_frame = 0
            for f in range(len(closest_frame_ranking)):
                if os.path.exists(self.images_dir / 'instance_segmentation_output' / ('%.6d.png' % closest_frame_ranking[f])):
                    closest_frame = closest_frame_ranking[f]
                    break
            step += 1

            focus_hit_pt_i = np.asarray(self.awareness_df["FocusInfo_HitPoint"][closest_frame])
            loc = self.awareness_df["EgoVariables_VehicleLoc"][closest_frame]
            loc = np.asarray([loc[0], loc[1], loc[2]])
            rot = self.awareness_df["EgoVariables_VehicleRot"][closest_frame]
            rot = np.asarray([rot[0], rot[1], rot[2]])
            pts_mid, pts_left, pts_right = self.get_gaze_point(focus_hit_pt_i, loc, rot)
            raw_gaze_mid.append(pts_mid)
            raw_gaze_left.append(pts_left)
            raw_gaze_right.append(pts_right)
        gaze_heatmap = gaussian_contour_plot(np.array(rgb_image), raw_gaze_mid, sigma=self.gaussian_sigma , contour_levels=3)
        gaze_heatmap_left = gaussian_contour_plot(np.array(rgb_left_image), raw_gaze_left, sigma=self.gaussian_sigma , contour_levels=3)
        gaze_heatmap_right = gaussian_contour_plot(np.array(rgb_right_image), raw_gaze_right, sigma=self.gaussian_sigma , contour_levels=3)
        
        # Convert all images to tensors
        rgb_image = transforms.functional.to_tensor(rgb_image)
        rgb_left_image = transforms.functional.to_tensor(rgb_left_image)
        rgb_right_image = transforms.functional.to_tensor(rgb_right_image)
        instance_seg_image = transforms.functional.to_tensor(instance_seg_image)
        instance_seg_left_image = transforms.functional.to_tensor(instance_seg_left_image)
        instance_seg_right_image = transforms.functional.to_tensor(instance_seg_right_image)
        print(rgb_image.shape)
        print(instance_seg_image.shape)
        gaze_heatmap = transforms.functional.to_tensor(gaze_heatmap)
        print(gaze_heatmap.shape)
        gaze_heatmap_left = transforms.functional.to_tensor(gaze_heatmap_left)
        gaze_heatmap_right = transforms.functional.to_tensor(gaze_heatmap_right)


        validity = frame_filter(frame_num, self.awareness_df)

        input_images = (rgb_image, instance_seg_image, gaze_heatmap, 
                        rgb_left_image, instance_seg_left_image, gaze_heatmap_left, 
                        rgb_right_image, instance_seg_right_image, gaze_heatmap_right)
        final_input_image = torch.cat(input_images)

        return final_input_image, label_mask_image, validity
    
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





        