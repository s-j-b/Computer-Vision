#This program calibrates the projector-camera setup

import cv2
import numpy as np
from glob import glob

#This function finds the corners in the camera calib images and returns height, width, and
# object and cam-frame points
def getCamData(img_names):
    
    #dimensions of inner chessboard
    patternSize = (9,6)
    
    #dimensions of a square in meters
    squareSize = .0245
    
    numSquares = patternSize[0]*patternSize[1]
    
    #printed points in the object frame
    squareCornersList = []
    
    #printed points in the camera frame
    warpedCornersList = []
    
    #Get object frame points
    xc = np.arange(patternSize[0], dtype='float32')*squareSize
    yc = np.arange(patternSize[1], dtype='float32')*squareSize
    x, y = np.meshgrid(xc, yc)
    
    #go from 2D arry to 1D lists
    x = x.flatten().astype('float32')
    y = y.flatten().astype('float32')
    z = np.zeros_like(x)
    
    #the final list of object frame points
    squareCorners = np.vstack( (x,y,z) ).transpose().reshape((-1, 1, 3))
    
    vims = []
    for i in range(len(img_names)):
        
        #get calibration image
        img = cv2.imread(img_names[i],0)
        
        #find those chessboard corners
        ret, warpedCorners = cv2.findChessboardCorners(img,
                                                       patternSize,
                                                       flags = cv2.CALIB_CB_ADAPTIVE_THRESH)
        
        #add the same square object frame corners to our list
        squareCornersList.append(squareCorners.astype('float32'))

        #add the camera frame corners to our list
        warpedCornersList.append(warpedCorners.astype('float32'))
        
        #the following 8 lines exist for testing purposes
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, patternSize, warpedCorners, ret)
        
        center = ( int(warpedCorners[-1,0,0]), int(warpedCorners[-1,0,1]) )
        center2 = ( int(warpedCorners[5,0,0]), int(warpedCorners[5,0,1]) )

        cv2.circle(vis, center, 10, (255, 0, 255), 2, cv2.CV_AA)
        cv2.circle(vis, center2, 10, (255, 0, 255), 2, cv2.CV_AA)
    
        vims.append(vis)
    
    ###
    ### for testing
    ###
    #    for i in range(len(vims)):
    #
    #        pic = vims[i]
    #        cv2.imshow('win', pic)
    #        cv2.waitKey()

    hC,wC = img.shape

    #return image dimensons and both vectors of vectors of points
    return (hC,wC,squareCornersList,warpedCornersList)

#This function obtains a plane homography mapping one space to another
#   -Takes in K, R, and t
def plane_homog_from_KRt(K, R, t):
    
    '''Construct a homography from a plane with given rotation
        and point t to the camera sensor plane.'''
    assert(K.shape == (3,3))
    assert(R.shape == (3,3))
    assert(t.shape == (3,))
    Rx = R[:,0]
    Ry = R[:,1]
    P = np.vstack((Rx, Ry, t)).transpose()
    
    return np.dot(K, P)

#Calibrate the camera
def camCalib(hC,wC,squareCornersList,warpedCornersList):

    #Run calibrate camera on those points from earlier and get a bunch of useful data
    # including K_cam and the rvecs and tvecs (rotation and translation vectors)
    camError, K_cam, dists_cam, rvecs, tvecs = cv2.calibrateCamera(squareCornersList,
                                                               warpedCornersList,
                                                               (wC,hC))

    #Manipulate those vectors so they're useful
    rvecs = tuple(rvecs)
    rvecs = np.concatenate(rvecs,1)
    rvecs = np.transpose(rvecs)

    tvecs = tuple(tvecs)
    tvecs = np.concatenate(tvecs,1)
    tvecs = np.transpose(tvecs)

    return (K_cam,dists_cam,rvecs,tvecs)

#Get the homography that maps the object frame points to the camera frame and vice-versa
def getHomogsOC(rvecs,tvecs,K_cam):
    
    # Each rvec, tvec pair is a rigid transformation between the
    # posterboard and the camera frame.  For instance, tvec[i] is the
    # camera-frame position of the center of the posterboard in image i
    # (and rvec[i] is the corresponding rotation of the board).
    i=0
    HsFromO = []
    HsFromC = []
    
    #get a homography for every image
    for i in range(len(rvecs)):
        
        rvec = rvecs[i]
        tvec = tvecs[i].flatten()
        
        # Rodrigues converts a rotation vector to a 3x3 rotation matrix
        # (or vice versa)
        R = cv2.Rodrigues(rvec)[0]

        # Construct a homography from the object (plane) to the camera
        # image sensor
        H_O2C = plane_homog_from_KRt(K_cam, R, tvec)
        
        # Construct the inverse homography as well
        H_C2O = np.linalg.inv(H_O2C)

        HsFromO.append(H_O2C)
        HsFromC.append(H_C2O)

    return (HsFromO,HsFromC)


#Get the projector-frame chessboard corners, along with some dimensions
def getProjData(n):
    chessboard = cv2.imread("chessboard.png")

    chessboard = cv2.resize(chessboard,(1200,750))

    ret, corners = cv2.findChessboardCorners(chessboard, (9,6))
    
    hP, wP, pD = np.shape(chessboard)
    
    cornersList = []
    for i in range(n):
        cornersList.append(corners)
    
    return (cornersList,(hP,wP))

#Run the final projector-camera calibration
def projCalib(hP, wP, cam_size, K_cam, dists_cam, img_names, HsFromC, projCornersProjList):
    
    patternSize = (9,6)

    projCornersObjList = []
    projCornersCamList = []
    vims = []

    #for every image, get these projected point sets
    for i in range(len(img_names)):

        #Read the justProj image
        img = cv2.imread(img_names[i],0)

        #Get its corners in the camera frame
        ret, projCornersCam = cv2.findChessboardCorners(img, patternSize,
                                                        flags = cv2.CALIB_CB_ADAPTIVE_THRESH)
        
        #for visualization later
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, patternSize, projCornersCam, ret)
        
        center = ( int(projCornersCam[-1,0,0]), int(projCornersCam[-1,0,1]) )
        center2 = ( int(projCornersCam[5,0,0]), int(projCornersCam[5,0,1]) )
        
        cv2.circle(vis, center, 10, (255, 0, 255), 2, cv2.CV_AA)
        cv2.circle(vis, center2, 10, (255, 0, 255), 2, cv2.CV_AA)
        
        vims.append(vis)
    
        #map camera frame images to object frame
        projCornersObj2D = cv2.perspectiveTransform(projCornersCam, HsFromC[i])
        
        #3Dify that biz
        projCornersObj = np.dstack( (projCornersObj2D,
                                     np.zeros((len(projCornersObj2D), 1, 1))))

        #Make a master list of object-frame projected corners
        projCornersObjList.append(projCornersObj.astype('float32'))
    
        #Make a master list of camera-frame projected corners
        projCornersCamList.append(projCornersCam.astype('float32'))


    ###
    ### for testing
    ###
    #    for i in range(len(vims)):
    #        pic = vims[i]
    #        cv2.imshow('win', pic)
    #        cv2.waitKey()
    #
    
    #Calibrate the projector, just like we did earlier with the camera
    projError, K_proj, dists_proj, rvecs, tvecs = cv2.calibrateCamera(projCornersObjList,
                                                                 projCornersProjList,
                                                                 (wP,hP))
                                                                
    #Run stereocalibrate using the given intrinsic matrices and distortion coefficients
    #This will return the desired R and t from the camera to the projector
    rval, Kc2, dc2, Kp2, dp2, R, t, E, F = cv2.stereoCalibrate(projCornersObjList,
                                                               projCornersCamList,
                                                               projCornersProjList,
                                                               cam_size,
                                                               K_cam,
                                                               dists_cam,
                                                               K_proj,
                                                               dists_proj)
                                                               

    return (K_cam, K_proj, R, t, dists_cam)




"""
    Program description - Matt Zucker:
    
    Take n images. In each image, we can see two chessboards on the same
    plane: one printed onto the plane, and one projected from the projector.
    
    1) For each image, find the chessboard corners for the printed
    chessboard. You will also need to know the object-frame (i.e. relative
    to the corner of the poster board) coordinates of these chessboard
    corners. Send *all* of these n sets of point correspondences to
    cv2.calibrateCamera to obtain:
    
    - an intrinsic parameter matrix for the camera.
    
    - for each pose of the board, you get a rotation vector and
    translation vector as well (you will convert the rotation vector
    to a matrix with cv2.Rodrigues)
    
    2) Now for each image i:
    
    
    2a) Use the R, t pair along with the camera intrinsics to compute a
    homography which maps the sensor plane of the camera to the 2D
    object-frame coordinate system (i.e. the xy grid of the
    posterboard). (This is the thing that is labeled H_obj_from_cam
    below).
    
    2b) Now for each image i, find the chessboard corners for the
    projected chessboard. Use the homography to map their pixel
    locations on the sensor plane to the 2D plane (object frame)
    coordinates of each corner.
    
    3) You can now call cv2.calibrateCamera using the object points which
    are the 3D-ified things from step 2b, and the image points which are
    the original locations of the chessboard corners in the projected
    image (these should be on a regular grid in image space). In this
    program, the image points here are referred to by
    proj_img_points_proj.  From this you will get the intrinsic parameter
    matrix for the projector.
    
    4) Finally, you can call cv2.stereoCalibrate using the object points
    which again are the 3D-ified things from step 2b, the camera 1 points
    are the image sensor locations of the projected chessboard corners
    (proj_img_points_cam in this program), the camera 2 points are the
    projector plane locations of the points (proj_img_points_proj). This will
    hand you back the extrinsic parameters of the projector R, t.
    
"""
def main():
    #get image names
    paper_names = glob("justPaper/*.jpg")
    proj_names = glob("justProj/*.jpg")
    
    n = len(paper_names)
    
    # Get camera data
    hC,wC,squareCornersList,warpedCornersList = getCamData(paper_names)
    
    #Step 1: Calibrate camera
    K_cam,dists_cam,rvecs,tvecs = camCalib(hC,wC,squareCornersList,warpedCornersList)
    
    #Step 2a: Get homographies between object frame and camera frame
    HsFromO,HsFromC = getHomogsOC(rvecs,tvecs,K_cam)
    
    #Step 2b: Get projector data and projector-frame points
    projCornersProjList,(hP,wP) = getProjData(n)
    
    #Step 3&4: Calibrate projector and run stereocalibrate to get R and t
    data = projCalib(hP, wP, (wC,hC), K_cam, dists_cam, proj_names, HsFromC,
                     projCornersProjList)
    
    #Return that data
    return data

calibData = main()
