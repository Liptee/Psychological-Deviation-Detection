from gaze_tracking.main import gaze_tracker

results = gaze_tracker(0, vectors=False, distances=True, frames=50)
print(results)