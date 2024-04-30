from dataset_full_corrected import SituationalAwarenessDataset
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# awareness_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/results/exp_allan_51-awdata.json"
# images_dir = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_allan-51_02_20_2024_17_21_58/images"
# awareness_df = pd.read_json(awareness_parse_file, orient='index')
sensor_config_file = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/sensor_config.ini"
raw_data = "/home/srkhuran-local/raw_data_corrected"


#sitawdata = SituationalAwarenessDataset(images_dir, awareness_df, sensor_config_file, gaussian_sigma = 10.0)
print("Creating dataset")
sitawdata = SituationalAwarenessDataset(raw_data, sensor_config_file, "yooni-21")

_, _, flow = sitawdata.__getitem__(51)