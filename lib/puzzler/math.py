import puzzler
import numpy as np

def unit_vector(v):
    return v / np.linalg.norm(v)
    
def distance_to_line(pt, line):
    # x0,y0 = pt
    # x1,y1 = line[0]
    # x2,y2 = line[1]
    #
    # d = ((x0-x1)*(y2-y1) - (y0-y1)*(x2-x1)) / hypot(x2-x1,y2-y1)
    vec1 = unit_vector(line[1] - line[0])
    vec2 = pt - line[0]
    return np.cross(vec2, vec1)

def vector_to_line(pt, line):
    vec1 = unit_vector(line[1] - line[0])
    vec2 = pt - line[0]
    dist = np.cross(vec2, vec1)
    # vector is perpendicular to the line
    return dist * np.array((-vec1[1], vec1[0]))

