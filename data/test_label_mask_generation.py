import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image


def read_corrected_csv(path):
    corrected_labels_df = pd.read_csv(path)
    # Extract columns into numpy arrays
    frame_no = corrected_labels_df['frame_no'].to_numpy()
    visible_total = corrected_labels_df['visible_total'].apply(eval).apply(np.array).to_numpy()
    awareness_label = corrected_labels_df['awareness_label'].apply(eval).apply(np.array).to_numpy()
    
    data = {'frame_no': frame_no,
    'visible_total': visible_total,
    'awareness_label': awareness_label}

    return pd.DataFrame(data)

def get_corrected_label_mask(inst_img, visible_ids, awareness_labels):
    width, height = inst_img.size
    mask = np.zeros((height, width), dtype=np.uint8)
    #mask = np.zeros((width, height))
    raw_img = np.array(inst_img)
    b = raw_img[:, :, 2]
    g = raw_img[:, :, 1]
    # Calculate the sum of b*256 + g
    sum_bg = (b * 256) + g
    
    for id_idx, id in enumerate(visible_ids):
        if awareness_labels[id_idx] == 1:
            # Create a mask where sum_bg is equal to target_value
            mask[sum_bg == id] = 100
        else:
            mask[sum_bg == id] = 200
    import ipdb; ipdb.set_trace()
    return mask

raw_data_dir = '/media/storage/raw_data_corrected'
episode = 'cbdr10-23'
images_dir = Path(os.path.join(raw_data_dir, episode, "images"))

frame_num = 6021
rgb_frame_delay = 3

instance_seg_image = Image.open(images_dir / 'instance_segmentation_output' / ('%.6d.png' % frame_num)).convert('RGB')
corrected_labels_df_filename = os.path.join(raw_data_dir, episode, 'corrected-awlabels.csv')
corrected_labels_df = read_corrected_csv(corrected_labels_df_filename)
visible_total = corrected_labels_df['visible_total'][frame_num- rgb_frame_delay - 1] # -1 because corrected df is one frame shifted from awareness df
awareness_label = corrected_labels_df['awareness_label'][frame_num- rgb_frame_delay - 1] # -1 because corrected df is one frame shifted from awareness df
full_label_mask = get_corrected_label_mask(instance_seg_image, visible_total, awareness_label)
import ipdb; ipdb.set_trace()
np.save('inst_seg_image.npy', instance_seg_image)
np.save('full_label_mask.npy', full_label_mask)