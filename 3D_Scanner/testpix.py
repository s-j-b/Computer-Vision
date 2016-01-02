import cv2
import numpy as np
from glob import glob


def isgood(img,patternSize):
    
    ret, warpedCorners = cv2.findChessboardCorners(img,
                                                   patternSize,
                                                   flags = cv2.CALIB_CB_ADAPTIVE_THRESH)
                                                   
                                                   
    #Unless the image has the correct number of corners, return failure
    if type(warpedCorners)!=type(None):
        if len(warpedCorners)==54:
            return True
    return False


def main():

    patternSize = (9,6)
    
    #get image names
    paper_names = glob("justPaper/*.jpg")
    proj_names = glob("justProj/*.jpg")
    
    #read images with projector masked, print the bad ones
    for name in paper_names:
        img = cv2.imread(name)
        if not isgood(img,patternSize):
            print name

    #read images with paper masked, print the bad ones
    for name in proj_names:
        img = cv2.imread(name)
        if not isgood(img,patternSize):
            print name


main()
