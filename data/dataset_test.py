from dataset_full import SituationalAwarenessDataset
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# awareness_parse_file = "/home/srkhuran-local/CarlaDReyeVR/DReyeVR-parser/results/exp_allan_51-awdata.json"
# images_dir = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/exp_allan-51_02_20_2024_17_21_58/images"
# awareness_df = pd.read_json(awareness_parse_file, orient='index')
sensor_config_file = "/home/srkhuran-local/CarlaDReyeVR/carla/PythonAPI/examples/sensor_config.ini"
raw_data = "/home/srkhuran-local/raw_data"


#sitawdata = SituationalAwarenessDataset(images_dir, awareness_df, sensor_config_file, gaussian_sigma = 10.0)
print("Creating dataset")
sitawdata = SituationalAwarenessDataset(raw_data, sensor_config_file, "cbdr10-36")

final_concat_image, label_mask_image, validity = sitawdata.__getitem__(51)


print(final_concat_image.shape)
print(validity)