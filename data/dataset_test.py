from dataset_full import SituationalAwarenessDataset
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

awareness_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/results/exp_allan_51-awdata.json"
images_dir = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_allan-51_02_20_2024_17_21_58/images"
awareness_df = pd.read_json(awareness_parse_file, orient='index')
sensor_config_file = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/sensor_config.ini"
recording_path = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/recording_files/exp_allan-51_02_20_2024_17_21_58.rec"

sitawdata = SituationalAwarenessDataset(recording_path, images_dir, awareness_df, sensor_config_file, gaussian_sigma = 10.0)

rgb_image, instance_seg_image, gaze_heatmap, rgb_left_image, instance_seg_left_image, gaze_heatmap_left, rgb_right_image, instance_seg_right_image, gaze_heatmap_right, gaze_heatmap, validity = sitawdata.__getitem__(1525)

rgb_image.show()
instance_seg_image.show()
gaze_heatmap.show()
gaze_heatmap_left.show()
gaze_heatmap_right.show()
print(validity)