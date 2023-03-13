from tools.record_data import record_data
from tools.train import autoencoder, autoencode_timelaps
from tools.realtime_detect import realtime_detect, realtime_detect_in_timelaps, realtime_anomaly_detect, anomaly_rowtime
from tools.utils.saver import load_data

PATH_TO_DATA = "anomaly_data"
CAM_LIST = ("CAM1", "CAM2", "IR")
OBJECTS = ("Andrey", "Artyom", "Pirog", "Vladimir")


# for record data
# for cam in CAM_LIST:
#     for object in OBJECTS:
#         X = load_data(f"{PATH_TO_DATA}/{cam}/{object}/normal", "mp4")
#         for x in X:
#             output = x.split(".")[0]
#             output = f"{output}.csv"
#             record_data(output_file=output,
#                         num_frames=1000,
#                         pose_landmarks=True,
#                         pose_cut=True,
#                         class_name="Normal",
#                         source=x)


# for autoencode
# for cam in CAM_LIST:
#     for object in OBJECTS:
#         X = load_data(f"{PATH_TO_DATA}/{cam}/{object}/normal", "csv")
#         model_name = f"{PATH_TO_DATA}/{cam}/{object}/model.pkl"
#         counter = 0
#         for x in X:
#             if counter == 0:
#                 autoencoder(data_file=x,
#                             epochs=500,
#                             train_size=0.9,
#                             learning_rate=3e-4,
#                             batch_size=32,
#                             output_file=model_name)
#             else:
#                 autoencoder(data_file=x,
#                             epochs=500,
#                             train_size=0.9,
#                             learning_rate=3e-4,
#                             batch_size=32,
#                             model_name=model_name,
#                             output_file=model_name)
#
#             counter += 1

# for timelaps autoencode
for cam in CAM_LIST:
    for object in OBJECTS:
        X = load_data(f"{PATH_TO_DATA}/{cam}/{object}/normal", "csv")
        model_name = f"{PATH_TO_DATA}/{cam}/{object}/model_time.pkl"
        counter = 0
        for x in X:
            if counter == 0:
                autoencode_timelaps(data_file=x,
                                    output_file=model_name,
                                    epochs=500,
                                    train_size=0.9,
                                    learning_rate=3e-4,
                                    batch_size=32,
                                    num_neighboor_frames=[-5, -3, -1])
            else:
                autoencode_timelaps(data_file=x,
                                    output_file=model_name,
                                    epochs=500,
                                    model_name=model_name,
                                    train_size=0.9,
                                    learning_rate=3e-4,
                                    batch_size=32,
                                    num_neighboor_frames=[-5, -3, -1])

            counter += 1