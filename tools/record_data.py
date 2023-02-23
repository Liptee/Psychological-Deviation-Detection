import cv2
import mediapipe
from tqdm import tqdm

import tools.utils.settings as settings
from tools.utils.drawing import drawing_on_frame
from tools.utils.saver import write_line_in_csv, add_data_in_row, init_csv_file
from tools.utils.back import return_num_params

mp_drawing = mediapipe.solutions.drawing_utils
mp_holistic = mediapipe.solutions.holistic

def record_data(output_file: str, num_frames: int, pose_landmarks: bool, face_landmarks: bool, left_hand_landmarks: bool, right_hand_landmarks: bool, pose_cut: bool = False):
    """
        This function extracts from webcam frames Action Units and 
        write them into csv file. Also this code require to label frame for every 
        launch.

        Parameters:
            output_file: str
            Name for resulting csv file.

            num_frames: int
            How much samples of data you wanna create in csv file.
            It doesn't tell function how long continue record, but how much
            samples of correct data will be in csv file

            pose_landmarks: bool
            Write pose landmarks or don't

            face_landmarks: bool
            Write face landmarks or don't
            
            left_hand_landmarks: bool
            Write left hand landmarks or don't

            right_hand_landmarks: bool
            Write right hand landmarks or don't
    """
    num_params = return_num_params(pose_landmarks, face_landmarks, right_hand_landmarks, left_hand_landmarks, pose_cut)

    #check if the file exists
    try:
        with open(output_file, 'r') as file:
            pass
    except:
        init_csv_file(output_file, num_params)
    
    count = 0
    class_name = input("Enter the name of the class: ")
    with mp_holistic.Holistic() as holistic:
        cap = cv2.VideoCapture(0)
        pbar = tqdm(total=num_frames)
        while num_frames != count:
            _, frame = cap.read()
            results = holistic.process(frame)
            row = []
            if results.pose_landmarks and pose_landmarks:
                frame = drawing_on_frame(frame, results.pose_landmarks, (0, 255, 0), mp_holistic.POSE_CONNECTIONS)
                row = add_data_in_row(row, results.pose_landmarks.landmark)
                if pose_cut:
                    row = row[settings.FACE_PARAMS_IN_POSE:]

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
                count += 1
                pbar.update(1)

            cv2.imshow('Raw Webcam Feed', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break