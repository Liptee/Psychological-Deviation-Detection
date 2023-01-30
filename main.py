from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from tools.record_data import record_data
from tools.train import train
from tools.realtime_detect import realtime_detect


#record_data("guest.csv", num_frames=100, face_landmarks=False, right_hand_landmarks=True)

pipelines = {
    'RFC':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'GBC':make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

#train("guest.csv", pipelines, test=True)
realtime_detect("models/guest_GBC.pkl", right_hand_landmarks=True)