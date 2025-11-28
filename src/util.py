# src/util.py
import math

def length(a, b):
    (x1,y1),(x2,y2) = a,b
    return math.hypot(x2-x1, y2-y1)

def angle_of_line(x1,y1,x2,y2):
    return math.atan2((y2 - y1),(x2 - x1))

def normalize_angle(a):
    # map angle to [-pi/2, pi/2)
    while a <= -math.pi/2:
        a += math.pi
    while a > math.pi/2:
        a -= math.pi
    return a

def lines_distance(seg_a, seg_b):
    # seg_a and seg_b are sequences of coords or endpoints: ((x1,y1),(x2,y2))
    mx1 = (seg_a[0][0] + seg_a[1][0]) / 2.0
    my1 = (seg_a[0][1] + seg_a[1][1]) / 2.0
    mx2 = (seg_b[0][0] + seg_b[1][0]) / 2.0
    my2 = (seg_b[0][1] + seg_b[1][1]) / 2.0
    return math.hypot(mx2-mx1, my2-my1)
