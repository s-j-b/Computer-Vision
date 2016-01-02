import cv2
import numpy as np
"""
   Gets calibration images 
"""

#Take image, duh
def takeImage(cam, win, display):
    
    #While our cam is solid
    while( cam.isOpened() ):
        #snap a pic
        ret,scan = cam.read()
        
        #Show the chessboard
        cv2.imshow(win,display)
        
        #Check for escape attempt, otherwise "enter" saves a pic
        k = cv2.waitKey(10)
        if k == 13:
            return scan
        if k == 27:
            break




#Project chessboard onto object frame board
#Get images and save them to justPaper/ and justProj/
def main():
    w = 1200
    h = 750
    
    #Setup window
    win = "Window"
    cv2.namedWindow(win)
    cv2.moveWindow(win, 0, 0)
    
    #Prep the proj chessboard
    display = cv2.imread("chessboard.png")
    display = cv2.resize(display, (w,h))

    #Setup cam
    cam = cv2.VideoCapture(0)
    
    #Take up to 20 pictures
    for i in range(20):
        
        #get image
        scan = takeImage(cam, win, display)

        #save image
        fname = str(i) + ".jpg"
        paper = "justPaper/"
        proj = "justProj/"
        
        cv2.imwrite(paper+fname, scan)
        cv2.imwrite(proj+fname, scan)
        
        print i,i,i

main()

