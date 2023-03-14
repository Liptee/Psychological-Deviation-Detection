from tools.record_data import record_data
from tools.train import autoencoder, autoencode_timelaps, create_json_autoencoders
from tools.realtime_detect import realtime_anomaly_detect, anomaly_rowtime
from tools.utils.saver import load_data
from tools.utils.back import find_max_med

PATH_TO_DATA = "anomaly_data"
CAM_LIST = ("CAM1", "CAM2", "IR")
OBJECTS = ("Andrey", "Artyom", "Pirog", "Vladimir")
NUMS_NEIGBOOR = [-12, -10, -8, -6, -4,-2]


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

# # for timelaps autoencode
# for cam in CAM_LIST:
#     for object in OBJECTS:
#         X = load_data(f"{PATH_TO_DATA}/{cam}/{object}/normal", "csv")
#         model_name = f"{PATH_TO_DATA}/{cam}/{object}/model_time.pkl"
#         counter = 0
#         for x in X:
#             if counter == 0:
#                 autoencode_timelaps(data_file=x,
#                                     output_file=model_name,
#                                     epochs=500,
#                                     train_size=0.9,
#                                     learning_rate=3e-4,
#                                     batch_size=32,
#                                     num_neighboor_frames=NUMS_NEIGBOOR)
#             else:
#                 autoencode_timelaps(data_file=x,
#                                     output_file=model_name,
#                                     epochs=500,
#                                     model_name=model_name,
#                                     train_size=0.9,
#                                     learning_rate=3e-4,
#                                     batch_size=32,
#                                     num_neighboor_frames=NUMS_NEIGBOOR)
#
#             counter += 1
#
#         create_json_autoencoders(model_file=model_name,
#                                  dir=f"{PATH_TO_DATA}/{cam}/{object}/normal",
#                                  output_data=f"{PATH_TO_DATA}/{cam}/{object}/model_time.json",
#                                  pose_landmarks=True,
#                                  pose_cut=True,
#                                  num_neighboor=NUMS_NEIGBOOR)


# for timelaps anomaly detect
for cam in CAM_LIST:
    for object in OBJECTS:
        X = load_data(f"{PATH_TO_DATA}/{cam}/{object}/normal", "mp4")
        model_name = f"{PATH_TO_DATA}/{cam}/{object}/model_time.pkl"
        for x in X:
            anomaly_rowtime(model_name,
                            source=x,
                            path_to_metadata=f"{PATH_TO_DATA}/{cam}/{object}/model_time.json",
                            func_to_coef=find_max_med,
                            pose_landmarks=True,
                            pose_cut=True,
                            num_neighboor_frames=NUMS_NEIGBOOR)