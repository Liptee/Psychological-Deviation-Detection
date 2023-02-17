
from math import sqrt

def cosine_distance(a, b):
    if len(a) != len(b):
        raise ValueError("a and b must be same length")
    numerator = sum(tup[0] * tup[1] for tup in zip(a, b))
    denoma = sum(avalue ** 2 for avalue in a)
    denomb = sum(bvalue ** 2 for bvalue in b)
    result = 1 - numerator / (sqrt(denoma)*sqrt(denomb))
    return result