import cv2
import numpy as np

#This program helps align the object with the bounding box in the camera's frame
#We do this to remove other annoying light that will mess up our data

def takeImage(cam, win, display, w, h):
    
    while( cam.isOpened() ):
        ret,scan = cam.read()
        
        mask = np.zeros_like(scan)
    
        #mask image
        mask[360:750,700:1100]=(1,1,1)
    
        scan = mask*scan
        
        #Show masked image
        cv2.imshow(win,scan)
        cv2.waitKey()
        
        k = cv2.waitKey(10)
        if k == 13:

            return scan
        if k == 27:
            break





def main():
    w = 1200
    h = 730

    win = "Window"
    cv2.namedWindow(win)

    display = cv2.imread("chessboard.png")

    display = cv2.resize(display, (w,h))

    cam = cv2.VideoCapture(0)
    
    #run test
    for i in range(20):
        scan = takeImage(cam, win, display, w, h)

main()

