import cv2
import mediapipe

import tools.utils.settings as settings
from tools.utils.drawing import drawing_on_frame
from tools.utils.saver import write_line_in_csv, add_data_in_row, init_csv_file

mp_drawing = mediapipe.solutions.drawing_utils
mp_holistic = mediapipe.solutions.holistic

def record_data(output_file, num_frames=100, pose_landmarks=False, face_landmarks=False, left_hand_landmarks=False, right_hand_landmarks=False):
    num_params = 0
    if pose_landmarks:
        num_params += settings.POSE_PARAMS
    if face_landmarks:
        num_params += settings.FACE_PARAMS
    if left_hand_landmarks:
        num_params += settings.HAND_PARAMS
    if right_hand_landmarks:
        num_params += settings.HAND_PARAMS

    #check if the file exists
    try:
        with open(output_file, 'r') as file:
            pass
    except:
        init_csv_file(output_file, num_params)

    class_name = input("Enter the name of the class: ")
    with mp_holistic.Holistic() as holistic:
        cap = cv2.VideoCapture(0)
        while num_frames >= 0:
            _, frame = cap.read()
            results = holistic.process(frame)
            row = []
            if results.pose_landmarks and pose_landmarks:
                frame = drawing_on_frame(frame, results.pose_landmarks, (0, 255, 0), mp_holistic.POSE_CONNECTIONS)
                row = add_data_in_row(row, results.pose_landmarks.landmark)

            if results.face_landmarks and face_landmarks:
                frame = drawing_on_frame(frame, results.face_landmarks, (255, 0, 0), mp_holistic.FACEMESH_TESSELATION)
                row = add_data_in_row(row, results.face_landmarks.landmark)

            if results.left_hand_landmarks and left_hand_landmarks:
                frame = drawing_on_frame(frame, results.left_hand_landmarks, (0, 0, 255), mp_holistic.HAND_CONNECTIONS)
                row = add_data_in_row(row, results.left_hand_landmarks.landmark)

            if results.right_hand_landmarks and right_hand_landmarks:
                frame = drawing_on_frame(frame, results.right_hand_landmarks, (0, 80, 120), mp_holistic.HAND_CONNECTIONS)
                row = add_data_in_row(row, results.right_hand_landmarks.landmark)
            
            if len(row) == num_params:
                row.insert(0, class_name)
                write_line_in_csv(output_file, row)
                num_frames -= 1

            cv2.imshow('Raw Webcam Feed', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break