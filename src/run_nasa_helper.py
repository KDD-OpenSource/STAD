from helper.nasa_helper import NasaHelper
import os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
nasa_raw_data_dir = os.path.join(root.parent, 'nasa_data')
output_dir = os.path.join(root, 'data', 'nasa')
label_file_path = os.path.join(root.parent, 'nasa_data', 'labeled_anomalies.csv')
default_win_len = 100

helper = NasaHelper(nasa_raw_data_dir, output_dir, label_file_path, default_win_len)
helper.load()
