import cv2
import numpy as np
from glob import glob

def main():
    data = glob("data*txt")
    
    
    f = open(data[0])
    
    pts = []
    xs = []
    l = f.readline()
    while l != '':
        vals = l.split(".")
        pts.append(np.array([int(vals[0]),int(vals[1]),1]))
        xs.append(int(vals[2]))
        l = f.readline()
    
    pts = np.array(pts)
    xs = list(set(xs))

    print pts
    print
    pts xs


main()
