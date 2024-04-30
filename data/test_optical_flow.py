import numpy as np
import os
import glob
from PIL import Image
import torch
from pathlib import Path
import pandas as pd
from optical_flow.pytorch_liteflownet.run import estimate
import flowiz as fz
import matplotlib.pyplot as plt
# Optical Flow generation

raw_data_dir = '/media/storage/raw_data_corrected'
episode = "yooni-21"
images_dir =  Path(os.path.join(raw_data_dir, episode, "images"))
aw_df_file_name = os.path.join(raw_data_dir, episode, 'rec_parse-awdata.json')
awareness_df = pd.read_json(aw_df_file_name, orient='index')
rgb_frame_delay = 3
frame_num = 1805



current_time = awareness_df["TimeElapsed"][frame_num - rgb_frame_delay]
end_time = end_time = current_time - 10
history_frames = []
history_frame_times = []
i = 0
frame = frame_num - i - 1
while frame > 0 and awareness_df["TimeElapsed"][frame - rgb_frame_delay] > end_time:
    history_frames.append(frame)
    history_frame_times.append(awareness_df["TimeElapsed"][frame - rgb_frame_delay])
    frame = frame_num - i - 1
    i+=1
prev_time = current_time - 0.4
closest_frame2_indices_ranking = np.argsort(abs(np.array(history_frame_times)-prev_time))
closest_frame2_ranking = [history_frames[i] for i in closest_frame2_indices_ranking]
    
closest_frame2 = 0
for f in range(len(closest_frame2_ranking)):
    f_num = closest_frame2_ranking[f] + rgb_frame_delay
    if os.path.exists(images_dir / 'rgb_output' / ('%.6d.png' % f_num)):
        closest_frame2 = closest_frame2_ranking[f]
        break
img1_path = images_dir / 'rgb_output' / ('%.6d.png' % frame_num)
im2_f_num = closest_frame2 + rgb_frame_delay
img2_path = images_dir / 'rgb_output' / ('%.6d.png' % im2_f_num)
tenOne = torch.FloatTensor(np.ascontiguousarray(np.array(Image.open(img1_path).convert('RGB'))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
tenTwo = torch.FloatTensor(np.ascontiguousarray(np.array(Image.open(img2_path).convert('RGB'))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

tenOutput = estimate(tenOne, tenTwo)

if not os.path.exists(images_dir / 'flo'):
    os.makedirs(images_dir / 'flo')
objOutput = open(images_dir/ 'flo' / ('%.6d_flow.flo' % frame_num), 'wb')

np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objOutput)
np.array([ tenOutput.shape[2], tenOutput.shape[1] ], np.int32).tofile(objOutput)
np.array(tenOutput.numpy().transpose(1, 2, 0), np.float32).tofile(objOutput)

objOutput.close()
flo_file = images_dir/ 'flo' / ('%.6d_flow.flo' % frame_num)
flow_img = fz.convert_from_file(str(flo_file))
plt.imshow(flow_img)
plt.show()