import csv
import numpy as np

def write_line_in_csv(output_file, data):
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data)


def add_data_in_row(row, results):
    row += list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results]).flatten())
    return row

def init_csv_file(output_file, num_params):
    landmarks = ['class']
    for val in range(1, num_params + 1):
        landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']

        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(landmarks)