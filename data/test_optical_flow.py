import numpy as np
import os
import glob
from PIL import Image
import torch
from pathlib import Path
import pandas as pd
# Need to clone https://github.com/sniklaus/pytorch-liteflownet into optical_flow directory before running this script!
from optical_flow.pytorch_liteflownet.run import estimate
import flowiz as fz
import matplotlib.pyplot as plt

# Optical Flow generation
# Given a frame number, function determines that closest frame 0.1 secs previous to given frame
# From these two frames, the liteflownet optical flow is determine and converted into an image

def get_flow_image(awareness_df, images_dir, rgb_frame_delay, frame_num):
    current_time = awareness_df["TimeElapsed"][frame_num - rgb_frame_delay]
    end_time = current_time - 5
    history_frames = []
    history_frame_times = []
    i = 0
    frame = frame_num - i - 1
    
    # construct list of frames and frame timestep over a history of 5 secs
    while frame > 0 and awareness_df["TimeElapsed"][frame - rgb_frame_delay] > end_time:
        history_frames.append(frame)
        history_frame_times.append(awareness_df["TimeElapsed"][frame - rgb_frame_delay])
        frame = frame_num - i - 1
        i+=1
    
    # Construct a ranking of time differences of the frames in history and the previous time step target
    prev_time = current_time - 0.1
    closest_frame2_indices_ranking = np.argsort(abs(np.array(history_frame_times)-prev_time))
    closest_frame2_ranking = [history_frames[i] for i in closest_frame2_indices_ranking]
    
    # Determine frame, with time step closest to target (that is not the same as the current frame)
    closest_frame2 = 0
    for f in range(len(closest_frame2_ranking)):
        f_num = closest_frame2_ranking[f] + rgb_frame_delay
        if f_num != frame_num:
            if os.path.exists(images_dir / 'rgb_output' / ('%.6d.png' % f_num)):
                closest_frame2 = closest_frame2_ranking[f]
                break
    # Given the two frame numbers, get their rgb image paths
    print(frame_num)
    img1_path = images_dir / 'rgb_output' / ('%.6d.png' % frame_num)
    im2_f_num = closest_frame2 + rgb_frame_delay
    print(im2_f_num)
    img2_path = images_dir / 'rgb_output' / ('%.6d.png' % im2_f_num)
    
    #perform image preprocessing
    tenOne = torch.FloatTensor(np.ascontiguousarray(np.array(Image.open(img1_path).convert('RGB'))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    tenTwo = torch.FloatTensor(np.ascontiguousarray(np.array(Image.open(img2_path).convert('RGB'))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

    #Liteflow Net Flo computation
    tenOutput = estimate(tenOne, tenTwo)
    
    #Flo post processing
    if not os.path.exists(images_dir / 'flo'):
        os.makedirs(images_dir / 'flo')
    objOutput = open(images_dir/ 'flo' / ('%.6d_flow.flo' % frame_num), 'wb')

    np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objOutput)
    np.array([ tenOutput.shape[2], tenOutput.shape[1] ], np.int32).tofile(objOutput)
    np.array(tenOutput.numpy().transpose(1, 2, 0), np.float32).tofile(objOutput)

    if not os.path.exists(images_dir / 'flow_output'):
        os.makedirs(images_dir / 'flow_output')
    objOutput.close()
    
    # Convert flo file ot image
    flo_file = images_dir/ 'flo' / ('%.6d_flow.flo' % frame_num)
    flow_img_array = fz.convert_from_file(str(flo_file))
    flow_img = Image.fromarray(flow_img_array)
    flow_img = flow_img.save(images_dir/ 'flow_output' / ('%.6d.png' % frame_num)) 
    


raw_data_dir = '/media/storage/raw_data_corrected'
episode = "yooni-51"
images_dir =  Path(os.path.join(raw_data_dir, episode, "images"))
aw_df_file_name = os.path.join(raw_data_dir, episode, 'rec_parse-awdata.json')
awareness_df = pd.read_json(aw_df_file_name, orient='index')
rgb_frame_delay = 3

for i in range(100, len(os.listdir(Path(os.path.join(images_dir, "rgb_output"))))):
    image_name = sorted(os.listdir(Path(os.path.join(images_dir, "rgb_output"))))[i]
    frame_num = int(image_name.split(".")[0])
    print(frame_num)
    get_flow_image(awareness_df, images_dir, rgb_frame_delay, frame_num)
