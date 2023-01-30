import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils

def drawing_on_frame(frame, result, color, connection):
    mp_drawing.draw_landmarks(frame, 
    result,
    connection, 
    mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=2))
    return frame

def drawing_predict(frame: np.array, predict: str):
    cv2.rectangle(
        frame, 
        (0,0), 
        (250, 60), 
        (245,117,16), 
        -1)


    cv2.putText(
        frame, 
        'CLASS', 
        (95,12), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (0,0,0), 
        1, 
        cv2.LINE_AA)

    cv2.putText(
        frame, 
        predict.split(' ')[0], 
        (90, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA)
    return frame