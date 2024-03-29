import os
import csv
import numpy as np

from glob import glob


def write_line_in_csv(output_file, data):
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data)


def add_data_in_row(row, results):
    row += list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results]).flatten())
    return row


def init_csv_file(output_file, num_params):
    landmarks = ['class']
    for val in range(1, num_params//4 + 1):
        landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']

        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(landmarks)


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path: str, pattern: str):
    return glob(os.path.join(path, f"*{pattern}"))