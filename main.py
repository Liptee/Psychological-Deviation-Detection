from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from tools.record_data import record_data
from tools.train import train


# record_data("emotion", num_frames=200, face_landmarks=True)

# pipelines = {
#     'RFC':make_pipeline(StandardScaler(), RandomForestClassifier()),
#     'GBC':make_pipeline(StandardScaler(), GradientBoostingClassifier())
# }

# train("emotion.csv", pipelines, test=True)