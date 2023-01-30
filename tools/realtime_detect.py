import cv2
import pickle
import mediapipe
import numpy as np
import pandas as pd
import tools.utils.settings as settings
from tools.utils.drawing import drawing_predict
from tools.utils.saver import add_data_in_row

mp_drawing = mediapipe.solutions.drawing_utils
mp_holistic = mediapipe.solutions.holistic

def realtime_detect(model_file, pose_landmarks=False, face_landmarks=False, left_hand_landmarks=False, right_hand_landmarks=False):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    num_params = 0
    if pose_landmarks:
        num_params += settings.POSE_PARAMS
    if face_landmarks:
        num_params += settings.FACE_PARAMS
    if left_hand_landmarks:
        num_params += settings.HAND_PARAMS
    if right_hand_landmarks:
        num_params += settings.HAND_PARAMS

    with mp_holistic.Holistic() as holistic:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            _, frame = cap.read()
            results = holistic.process(frame)
            row = []
            if results.pose_landmarks and pose_landmarks:
                row = add_data_in_row(row, results.pose_landmarks.landmark)

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
