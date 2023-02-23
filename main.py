from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from tools.record_data import record_data
from tools.train import train, train_timelaps, autoencoder, autoencode_timelaps
from tools.realtime_detect import realtime_detect, realtime_detect_in_timelaps, realtime_anomaly_detect, anomaly_rowtime

memory = {}
command = ""
while command != "q":
    command = input("Enter a command: ")
    if command == "record":
        filename = input("Enter a filename: ")
        cut_pose = False
        if not filename in memory:
            memory[filename] = []
            if input("Face landmarks? (y/n): ") == "y":
                face_landmarks = True
            else: face_landmarks = False
            memory[filename].append(face_landmarks)
            if input("Pose landmarks? (y/n): ") == "y": 
                pose_landmarks = True
                if input("Do you want use cutting version? (y/n): ") == "y":
                    cut_pose = True
            else: pose_landmarks = False
            memory[filename].append(pose_landmarks)
            if input("right hand landmarks? (y/n): ") == "y":
                right_hand_landmarks = True
            else: right_hand_landmarks = False
            memory[filename].append(right_hand_landmarks)
            if input("left hand landmarks? (y/n): ") == "y":
                left_hand_landmarks = True
            else: left_hand_landmarks = False
            memory[filename].append(left_hand_landmarks)

        num_frames = int(input("Enter number of frames: "))
        record_data(f"{filename}.csv", num_frames=num_frames, face_landmarks=memory[filename][0], pose_landmarks=memory[filename][1], right_hand_landmarks=memory[filename][2], left_hand_landmarks=memory[filename][3], pose_cut=cut_pose)
# -------------------------------------------------------------------------------------------------------------------------

    if command == "train":
        filename = input("Enter a filename: ")
        pipelines = {
            'RFC':make_pipeline(StandardScaler(), RandomForestClassifier()),
            'GBC':make_pipeline(StandardScaler(), GradientBoostingClassifier())
        }
        test_size = float(input("Enter a test size: "))
        train(f"{filename}.csv", pipelines, test=True, test_size=test_size)
# -------------------------------------------------------------------------------------------------------------------------

    if command == "detect":
        model_name = input("Enter a model name: ")
        filename = model_name.split("_")[0]

        if not filename in memory:
            memory[filename] = []
            if input("Face landmarks? (y/n): ") == "y":
                face_landmarks = True
            else: face_landmarks = False
            memory[filename].append(face_landmarks)
            if input("Pose landmarks? (y/n): ") == "y": 
                pose_landmarks = True
            else: pose_landmarks = False
            memory[filename].append(pose_landmarks)
            if input("right hand landmarks? (y/n): ") == "y":
                right_hand_landmarks = True
            else: right_hand_landmarks = False
            memory[filename].append(right_hand_landmarks)
            if input("left hand landmarks? (y/n): ") == "y":
                left_hand_landmarks = True
            else: left_hand_landmarks = False
            memory[filename].append(left_hand_landmarks)
        
        realtime_detect(f"models/{model_name}.pkl", face_landmarks=memory[filename][0], pose_landmarks=memory[filename][1], right_hand_landmarks=memory[filename][2], left_hand_landmarks=memory[filename][3])
# -------------------------------------------------------------------------------------------------------------------------

    if command == "realtime_train":
        filename = input("Enter a filename: ")
        pipelines = {
            'RFC':make_pipeline(StandardScaler(), RandomForestClassifier()),
            'GBC':make_pipeline(StandardScaler(), GradientBoostingClassifier())
        }
        neighbors = []
        while True:
            neighbor = input("Enter number of neighbors: ")
            try:
                neighbors.append(int(neighbor))
            except:
                break
        test_size = float(input("Enter a test size: "))
        train_timelaps(f"{filename}.csv", pipelines, test=True, test_size=test_size, num_neighboor_frames=neighbors)
# -------------------------------------------------------------------------------------------------------------------------

    if command == "realtime_detect":
        model_name = input("Enter a model name: ")
        filename = model_name.split("_")[1]
        if not filename in memory:
            memory[filename] = []
            if input("Face landmarks? (y/n): ") == "y":
                face_landmarks = True
            else: face_landmarks = False
            memory[filename].append(face_landmarks)
            if input("Pose landmarks? (y/n): ") == "y": 
                pose_landmarks = True
            else: pose_landmarks = False
            memory[filename].append(pose_landmarks)
            if input("right hand landmarks? (y/n): ") == "y":
                right_hand_landmarks = True
            else: right_hand_landmarks = False
            memory[filename].append(right_hand_landmarks)
            if input("left hand landmarks? (y/n): ") == "y":
                left_hand_landmarks = True
            else: left_hand_landmarks = False
            memory[filename].append(left_hand_landmarks)

        neighbors = []
        while True:
            neighbor = input("Enter number of neighbors: ")
            try:
                neighbors.append(int(neighbor))
            except:
                break
        realtime_detect_in_timelaps(f"models/{model_name}.pkl", face_landmarks=memory[filename][0], pose_landmarks=memory[filename][1], right_hand_landmarks=memory[filename][2], left_hand_landmarks=memory[filename][3], num_neighboor_frames=neighbors)
# -------------------------------------------------------------------------------------------------------------------------

    if command == "autoencode":
        filename = input("Enter a filename: ")
        epochs = int(input("Input epochs: "))
        autoencoder(f"{filename}.csv", epochs)
# -------------------------------------------------------------------------------------------------------------------------

    if command == "anomaly":
        model_name = input("Enter a model name: ")
        filename = model_name.split("_")[0]
        cut_pose = False

        if not filename in memory:
            memory[filename] = []
            if input("Face landmarks? (y/n): ") == "y":
                face_landmarks = True
            else: face_landmarks = False
            memory[filename].append(face_landmarks)
            if input("Pose landmarks? (y/n): ") == "y": 
                pose_landmarks = True
                if input("Do you want use cutting version? (y/n): ") == "y":
                    cut_pose = True
            else: pose_landmarks = False
            memory[filename].append(pose_landmarks)
            if input("right hand landmarks? (y/n): ") == "y":
                right_hand_landmarks = True
            else: right_hand_landmarks = False
            memory[filename].append(right_hand_landmarks)
            if input("left hand landmarks? (y/n): ") == "y":
                left_hand_landmarks = True
            else: left_hand_landmarks = False
            memory[filename].append(left_hand_landmarks)
        
        realtime_anomaly_detect(f"models/{model_name}.pkl", face_landmarks=memory[filename][0], pose_landmarks=memory[filename][1], right_hand_landmarks=memory[filename][2], left_hand_landmarks=memory[filename][3], cut_pose=cut_pose)
# -------------------------------------------------------------------------------------------------------------------------
    
    if command == "timeraw autoencode":
        filename = input("Enter a filename: ")
        epochs = int(input("Input epochs: "))
        neighbors = []
        while True:
            neighbor = input("Enter number of neighbors: ")
            try:
                neighbors.append(int(neighbor))
            except:
                break
        train_size = float(input("train size: "))
        autoencode_timelaps(f"{filename}.csv", epochs, train_size=train_size, num_neighboor_frames=neighbors)
# -------------------------------------------------------------------------------------------------------------------------

    if command == "timeraw anomaly":
        model_name = input("Enter a model name: ")
        filename = model_name.split("_")[1]
        cut_pose = False

        if not filename in memory:
            memory[filename] = []
            if input("Face landmarks? (y/n): ") == "y":
                face_landmarks = True
            else: face_landmarks = False
            memory[filename].append(face_landmarks)
            if input("Pose landmarks? (y/n): ") == "y": 
                pose_landmarks = True
                if input("Do you want use cutting version? (y/n): ") == "y":
                    cut_pose = True
            else: pose_landmarks = False
            memory[filename].append(pose_landmarks)
            if input("right hand landmarks? (y/n): ") == "y":
                right_hand_landmarks = True
            else: right_hand_landmarks = False
            memory[filename].append(right_hand_landmarks)
            if input("left hand landmarks? (y/n): ") == "y":
                left_hand_landmarks = True
            else: left_hand_landmarks = False
            memory[filename].append(left_hand_landmarks)
        
        neighbors = []
        while True:
            neighbor = input("Enter number of neighbors: ")
            try:
                neighbors.append(int(neighbor))
            except:
                break
        anomaly_rowtime(f"models/{model_name}.pkl", face_landmarks=memory[filename][0], pose_landmarks=memory[filename][1], right_hand_landmarks=memory[filename][2], left_hand_landmarks=memory[filename][3], cut_pose=cut_pose, num_neighboor_frames=neighbors)

