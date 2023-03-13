from tools.record_data import record_data
from tools.train import train, train_timelaps, autoencoder, autoencode_timelaps
from tools.realtime_detect import realtime_detect, realtime_detect_in_timelaps, realtime_anomaly_detect, anomaly_rowtime
from tools.utils.saver import load_data

PATH_TO_DATA = "anomaly_data"
CAM_LIST = ("CAM1", "CAM2", "IR")
OBJECTS = ("Andrey", "Artyom", "Pirog", "Vladimir")

for cam in CAM_LIST:
    for object in OBJECTS:
        X = load_data(f"{PATH_TO_DATA}/{cam}/{object}/normal", "mp4")
        for x in X:
            output = x.split(".")[0]
            output = f"{output}.csv"
            record_data(output_file=output,
                        num_frames=1000,
                        pose_landmarks=True,
                        pose_cut=True,
                        class_name="Normal",
                        source=x)