from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from tools.record_data import record_data
from tools.train import train
from tools.realtime_detect import realtime_detect

memory = {}
command = ""
while command != "q":
    command = input("Enter a command: ")
    if command == "record":
        filename = input("Enter a filename: ")
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

        num_frames = int(input("Enter number of frames: "))
        record_data(f"{filename}.csv", num_frames=num_frames, face_landmarks=memory[filename][0], pose_landmarks=memory[filename][1], right_hand_landmarks=memory[filename][2], left_hand_landmarks=memory[filename][3])

    if command == "train":
        filename = input("Enter a filename: ")
        pipelines = {
            'RFC':make_pipeline(StandardScaler(), RandomForestClassifier()),
            'GBC':make_pipeline(StandardScaler(), GradientBoostingClassifier())
        }
        test_size = float(input("Enter a test size: "))
        train(f"{filename}.csv", pipelines, test=True, test_size=test_size)

    if command == "detect":
        model_name = input("Enter a model name: ")

        if input("Face landmarks? (y/n): ") == "y":
            face_landmarks = True
        else: face_landmarks = False
        if input("Pose landmarks? (y/n): ") == "y": 
            pose_landmarks = True
        else: pose_landmarks = False
        if input("right hand landmarks? (y/n): ") == "y":
            right_hand_landmarks = True
        else: right_hand_landmarks = False
        if input("left hand landmarks? (y/n): ") == "y":
            left_hand_landmarks = True
        else: left_hand_landmarks = False
        realtime_detect(f"models/{model_name}.pkl", face_landmarks=face_landmarks, pose_landmarks=pose_landmarks, right_hand_landmarks=right_hand_landmarks, left_hand_landmarks=left_hand_landmarks)