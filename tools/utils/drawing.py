import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

def drawing_on_frame(frame, result, color, connection):
    mp_drawing.draw_landmarks(frame, 
    result,
    connection, 
    mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=2))
    return frame