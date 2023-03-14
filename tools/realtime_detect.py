import cv2
import pickle
import json
import numpy as np
import mediapipe
import pandas as pd
import warnings
import tools.utils.settings as settings
from tools.utils.drawing import drawing_predict
from tools.utils.saver import add_data_in_row
from tools.utils.back import cosine_distance, return_num_params, find_max_avg

import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

mp_drawing = mediapipe.solutions.drawing_utils
mp_holistic = mediapipe.solutions.holistic


def find_max_med(data):
    max = 0.0
    for id in data:
        if data[id]['median'] > max:
            max = data[id]['average']
    print(max)
    return max

def realtime_detect(model_file, pose_landmarks=False, face_landmarks=False, left_hand_landmarks=False, right_hand_landmarks=False, cut_pose=False):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    num_params = return_num_params(pose_landmarks, face_landmarks, right_hand_landmarks, left_hand_landmarks, cut_pose)

    with mp_holistic.Holistic() as holistic:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            _, frame = cap.read()
            results = holistic.process(frame)
            row = []
            if results.pose_landmarks and pose_landmarks:
                row = add_data_in_row(row, results.pose_landmarks.landmark)
                if cut_pose:
                    row = row[settings.FACE_PARAMS_IN_POSE:]
            if results.face_landmarks and face_landmarks:
                row = add_data_in_row(row, results.face_landmarks.landmark)
     
            if results.left_hand_landmarks and left_hand_landmarks:
                row = add_data_in_row(row, results.left_hand_landmarks.landmark)       

            if results.right_hand_landmarks and right_hand_landmarks:
                row = add_data_in_row(row, results.right_hand_landmarks.landmark)

            
            if len(row) == num_params:
                X = pd.DataFrame([row])
                predict = model.predict(X)[0]
                frame = drawing_predict(frame, predict)

            cv2.imshow('Raw Webcam Feed', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
# -------------------------------------------------------------------------------------------------------------------------

def realtime_detect_in_timelaps(model_file: str, num_neighboor_frames: list = [-3, -1], pose_landmarks=False, face_landmarks=False, left_hand_landmarks=False, right_hand_landmarks=False, pose_cut=False):
    rows = []
    #calcucalte difference between min and max number of frames
    max_frames = max(num_neighboor_frames)
    min_frames = min(num_neighboor_frames)
    num_frames = max_frames - min_frames + 1 

    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    num_params = return_num_params(pose_landmarks, face_landmarks, right_hand_landmarks, left_hand_landmarks, pose_cut)

    num_params *= len(num_neighboor_frames) + 1
    with mp_holistic.Holistic() as holistic:
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            tmp_list = []
            _, frame = cap.read()
            results = holistic.process(frame)
            row = []
            if results.pose_landmarks and pose_landmarks:
                row = add_data_in_row(row, results.pose_landmarks.landmark)
                if pose_cut:
                    row = row[settings.FACE_PARAMS_IN_POSE:]
            if results.face_landmarks and face_landmarks:
                row = add_data_in_row(row, results.face_landmarks.landmark)
     
            if results.left_hand_landmarks and left_hand_landmarks:
                row = add_data_in_row(row, results.left_hand_landmarks.landmark)       

            if results.right_hand_landmarks and right_hand_landmarks:
                row = add_data_in_row(row, results.right_hand_landmarks.landmark)

            if len(rows) <= num_frames:
                rows.append(row)

            else:
                rows.pop(0)
                rows.append(row)
                tmp_list.extend(rows[-1])
                for i in reversed(num_neighboor_frames):
                    tmp_list.extend(rows[i-1])
                if len(tmp_list) == num_params:
                    X = pd.DataFrame([tmp_list])
                    predict = model.predict(X)[0]
                    frame = drawing_predict(frame, predict)
                    

            cv2.imshow('Raw Webcam Feed', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
# -------------------------------------------------------------------------------------------------------------------------

def realtime_anomaly_detect(model_file: str,
                            source = 0,
                            pose_landmarks=False,
                            face_landmarks=False,
                            left_hand_landmarks=False,
                            right_hand_landmarks=False,
                            pose_cut = False):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    num_params = return_num_params(pose_landmarks, face_landmarks, right_hand_landmarks, left_hand_landmarks, pose_cut)
    x = np.zeros((num_params))

    with mp_holistic.Holistic() as holistic:
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = holistic.process(frame)
            row = []
            if results.pose_landmarks and pose_landmarks:
                row = add_data_in_row(row, results.pose_landmarks.landmark)
                if pose_cut:
                    row = row[settings.FACE_PARAMS_IN_POSE:]

            if results.face_landmarks and face_landmarks:
                row = add_data_in_row(row, results.face_landmarks.landmark)
     
            if results.left_hand_landmarks and left_hand_landmarks:
                row = add_data_in_row(row, results.left_hand_landmarks.landmark)       

            if results.right_hand_landmarks and right_hand_landmarks:
                row = add_data_in_row(row, results.right_hand_landmarks.landmark)

            if len(row) == num_params:
                X = np.array(row)
                X = np.array([x, X])
                dataset = TensorDataset(torch.tensor(X.astype(np.float32)))
                data_loader = DataLoader(dataset, batch_size=32)
                preds = []
                for batch in data_loader:
                    pred = model(batch[0])
                    preds.append(pred.detach().numpy())
                orig = X[-1]
                pred = preds[-1][-1]
                state = "Normal"

                cosine_dist = cosine_distance(orig, pred)
                print(cosine_dist)
                if cosine_dist > 0.032:
                    state = "Anomaly"

                frame = drawing_predict(frame, state)

            cv2.imshow('Raw Webcam Feed', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
# -------------------------------------------------------------------------------------------------------------------------

def anomaly_rowtime(model_file: str,
                    source = 0,
                    path_to_metadata: str = None,
                    func_to_coef = find_max_avg,
                    pose_landmarks=False,
                    face_landmarks=False,
                    left_hand_landmarks=False,
                    right_hand_landmarks=False,
                    pose_cut = False,
                    num_neighboor_frames: list = [-3, -1],
                    show: bool = True,
                    return_data: bool = False):
    coef = 0.025
    if path_to_metadata:
        with open(path_to_metadata, 'r') as f:
            data = json.load(f)
        coef = func_to_coef(data)

    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    rows = []
    num_frames = abs(min(num_neighboor_frames)) 
    num_params = return_num_params(pose_landmarks, face_landmarks, right_hand_landmarks, left_hand_landmarks, pose_cut)

    num_params *= len(num_neighboor_frames) + 1
    x = np.zeros((num_params))

    print(model_file)
    print(source)

    dists = []

    with mp_holistic.Holistic() as holistic:
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            tmp_list = []
            ret, frame = cap.read()
            if not ret:
                if return_data:
                    data = {"min": np.min(dists),
                            "max": np.max(dists),
                            "average": np.average(dists),
                            "median": np.median(dists)}
                    return data
                break
            results = holistic.process(frame)
            row = []
            if results.pose_landmarks and pose_landmarks:
                row = add_data_in_row(row, results.pose_landmarks.landmark)
                if pose_cut:
                    row = row[settings.FACE_PARAMS_IN_POSE:]

            if results.face_landmarks and face_landmarks:
                row = add_data_in_row(row, results.face_landmarks.landmark)
     
            if results.left_hand_landmarks and left_hand_landmarks:
                row = add_data_in_row(row, results.left_hand_landmarks.landmark)       

            if results.right_hand_landmarks and right_hand_landmarks:
                row = add_data_in_row(row, results.right_hand_landmarks.landmark)

            if len(rows) <= num_frames:
                rows.append(row)

            else:
                rows.pop(0)
                rows.append(row)
                tmp_list.extend(rows[-1])
                for i in reversed(num_neighboor_frames):
                    tmp_list.extend(rows[i-1])
                if len(tmp_list) == num_params:
                    X = np.array(tmp_list)
                    X = np.array([x, X])
                    dataset = TensorDataset(torch.tensor(X.astype(np.float32)))
                    data_loader = DataLoader(dataset, batch_size=32)
                    preds = []
                    for batch in data_loader:
                        pred = model(batch[0])
                        preds.append(pred.detach().numpy())
                    orig = X[-1]
                    pred = preds[-1][-1]
                    state = "Normal"

                    cosine_dist = cosine_distance(orig, pred)
                    dists.append(cosine_dist)

                    if cosine_dist > coef:
                        state = "Anomaly"
                    
                    frame = drawing_predict(frame, state)
            if show:
                cv2.imshow('Raw Webcam Feed', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break