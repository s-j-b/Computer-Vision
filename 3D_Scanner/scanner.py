import cv2
import numpy as np
import numpy.ma as ma

#This function cleans an image using:
# - background subtraction
# - masking around the target box
# - looking only at green pixels
def clean(img, bg, win):
    
    #get difference of image from background
    diff = cv2.absdiff(img, bg)
    
    #get dims
    h,w,d = img.shape
    
    #make mask
    mask = np.zeros_like(img)
    
    #make mask only care about target box
    mask[360:750,700:1100]=(1,1,1)
    
    #apply that mask
    img = mask*diff

    #compare channels
    b,g,r = cv2.split(img)
    b = np.greater(g,b)
    r = np.greater(g,r)
    
    #Make a new 3-channel mask that ignores pixels that aren't mostly green
    channel = np.bitwise_and(b,r)
    off = np.array([channel,channel,channel], np.uint8)
    mask = cv2.merge(off)
    
    #apply the new mask!
    img = mask*img
    
    #black'n'white
    bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #binarize that biz
    final = cv2.threshold(bw, 100, 255, cv2.THRESH_BINARY)[1]
    
    return final

#Takes an image of the object-to-be-scanned
def takeImage(x, cam, win, scan, w, h):
    #copy background screen
    gamma = scan.copy()
    
    #draw cool green line
    cv2.line(gamma,(x,0),(x,h),(0,255,0),1)

    #show the line
    cv2.imshow(win,gamma)
    k = cv2.waitKey(1)
    
    #take that picture
    ret,scan = cam.read()

    return scan

#Gets nonzero points from a binarized image
def getPoints(img):
    
    #get x and y values for image
    yvals, xvals = np.where(img!=0)
    
    #add z component of 0
    zvals = np.ones(len(xvals))
    
    return np.transpose(np.vstack((xvals, yvals, zvals)))

#Run a single scan of the surface of the object
def scan(K,dists):
    
    #setup window and camera
    win = "scanner"
    cv2.namedWindow(win);
    cv2.moveWindow(win, 0, 0)
    cam = cv2.VideoCapture(0)
    
    h = 750
    w = 1200
    d = 3
    
    dims = (h,w,d)
    
    #Get a background image
    screen = np.zeros(dims, np.uint8)
    cv2.imshow(win,screen)
    cv2.waitKey()
    
    ret,bg = cam.read()
    
    scans = []

    #establish swath of projected line addresses
    Xs = range(350,1150,10)
    
    i=0
    
    
    imgs = []
    
    #Run the scan!
    for x in Xs:
        #Get image
        scan = takeImage(x, cam, win, screen, w, h)
        
        #Clean image to get green lines, then binarize
        bin = clean(scan, bg, win)

        #convert binarized image to points
        points = getPoints(bin)

        scans.append(points)
        imgs.append(bin)
    

    scans = np.asarray(scans)

    #return all the points and the xvalues of the lines
    return (w, h, scans, Xs, win)

