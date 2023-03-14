
from math import sqrt
import tools.utils.settings as settings

def cosine_distance(a, b):
    if len(a) != len(b):
        raise ValueError("a and b must be same length")
    numerator = sum(tup[0] * tup[1] for tup in zip(a, b))
    denoma = sum(avalue ** 2 for avalue in a)
    denomb = sum(bvalue ** 2 for bvalue in b)
    result = 1 - numerator / (sqrt(denoma)*sqrt(denomb))
    return result

def return_num_params(pose: bool,
                      face: bool,
                      right: bool,
                      left: bool,
                      cut_pose: bool) -> int:
    num_params = 0
    if pose:
        num_params += settings.POSE_PARAMS
    if face:
        num_params += settings.FACE_PARAMS
    if left:
        num_params += settings.HAND_PARAMS
    if right:
        num_params += settings.HAND_PARAMS
    if cut_pose and pose:
        num_params -= settings.FACE_PARAMS_IN_POSE
    return num_params


def find_max_med(data):
    max = 0.0
    for id in data:
        if data[id]['median'] > max:
            max = data[id]['average']
    print(max)
    return max


def find_max_avg(data):
    max = 0.0
    for id in data:
        if data[id]['average'] > max:
            max = data[id]['average']
    print(max)
    return max