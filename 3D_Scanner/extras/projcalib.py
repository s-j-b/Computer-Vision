'''Take n images. In each image, we can see two chessboards on the same
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

'''

import cv2
import numpy as np

def normalize(x):
    '''Returns a normalized (rescaled to unit length) copy of input vector.'''
    return x / np.linalg.norm(x)

def plane_to_3d_from_Rt(Rmat, tvec, ppts):

    '''Given a rotation matrix, a translation vector, and a set of points
    with shape (n,1,2) give 3D locations of points on plane.

    '''

    assert(Rmat.shape == (3,3))
    assert(len(ppts.shape) == 3 and 
           ppts.shape[1] == 1 and 
           ppts.shape[2] == 2)
    Rleft = Rmat[:,:2] # 3x2
    proj = np.dot(ppts.reshape((-1,2)), Rleft.T) 
    assert(proj.shape[1] == 3)
    proj += tvec.reshape((-1, 3))
    return proj.reshape((-1,1,3))

def plane_normal_from_R(Rmat):

    '''Given rotation matrix for a plane, extract the plane normal (last column)'''

    assert(Rmat.shape == (3,3))
    return Rmat[:,2]

def project_to_plane_from_np(ro, rd, n, p0):
    '''Given a ray origin, a set of ray directions of shape (n,1,3) and a
    plane normal and a point on the plane, intersect the bundle of
    rays with the plane in 3D.

    For synthesizing data only.

    '''
    assert(ro.shape == (3,))
    assert(len(rd.shape) == 3 and rd.shape[1] == 1 and rd.shape[2] == 3)
    rd_rect = rd.reshape((-1, 3))
    assert(n.shape == (3,))
    assert(p0.shape == (3,))
    # we have p = ro + rd*t
    # we also have p . n = p0 . n
    # hence (ro + rd*t) . n = p0 . n
    # and ro . n + t * rd.n = p0 . n
    # then t * (rd . n) = (p0 - r0) . n
    t = np.dot(p0 - ro, n) / np.dot(rd_rect, n)
    ppts = ro + t.reshape((-1,1)) * rd_rect
    return ppts.reshape((-1,1,3))

# get a plane homography which maps object(plane) coords to camera
# coords
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

def make_grid_2d(grid_size, square_size, offset):

    '''Create a mesh of 2D points from a grid size,
    square size, and lower-left origin.'''

    x, y = np.meshgrid(np.arange(grid_size[0]),
                       np.arange(grid_size[1]))

    x = x.flatten()
    y = y.flatten()

    xy = np.vstack((x, y)).transpose()

    xy = xy * square_size + offset

    return xy.reshape((-1, 1, 2))

######################################################################


def proj_calib(printed_obj_points,
               proj_img_points_proj,
               all_printed_img_points, 
               all_proj_img_points_cam,
               cam_size,
               proj_size):

    '''Run the projector calibration. Arguments:

      - printed_obj_points is an array of 3D points (all with zero
        z-coordinate) of shape (m,1,3)

      - proj_img_points_proj is an array of 2D points of shape (k,1,2)

      - all_printed_img_points are the detected locations of the
        printed chessboard in each image - a list of n arrays of shape
        (m,1,2)

      - all_proj_img_points_cam are the detected locations of the
        projected chessboard in each image - a list of n arrays of shape
        (k,1,2)

      - cam_size is a tuple of (cam_img_w, cam_img_h)

      - proj_size is a tuple of (proj_img_w, proj_img_h)
    
    Returns K_cam, K_proj, R, t, where the latter are intrinsics and
    extrinsics of projector.

    '''

    n_images = len(all_printed_img_points)
    assert( len(all_proj_img_points_cam) == n_images )

    n_printed = len(printed_obj_points)
    n_proj = len(proj_img_points_proj)

    # Set up flags to get simple camera calibrations (no distance coeffs)
    mono_flags = ( cv2.CALIB_ZERO_TANGENT_DIST |
                   cv2.CALIB_FIX_K1 |
                   cv2.CALIB_FIX_K2 |
                   cv2.CALIB_FIX_K3 |
                   cv2.CALIB_FIX_K4 |
                   cv2.CALIB_FIX_K5 |
                   cv2.CALIB_FIX_K6 )

    # Step 1: get camera intrinsics, as well as rotation/translation
    # vectors of the object in each image.

    all_printed_obj_points = []

    for i in range(n_images):
        all_printed_obj_points.append(printed_obj_points.astype('float32'))
    
    





    """
    print "Real pts of chessboard"
    print all_printed_obj_points

    print "$$$$$$"

    print
    print "Cam pts of chessboard"
    print all_printed_img_points

    """


    rval, K_cam, dists_cam, rvecs, tvecs = cv2.calibrateCamera(
        all_printed_obj_points, all_printed_img_points,
        cam_size, flags=mono_flags)

    all_proj_obj_points = []
    all_proj_img_points_proj = []
        
    # Step 2: get the object points
    for i in range(n_images):

        rvec = rvecs[i]
        tvec = tvecs[i].flatten()
        
        R = cv2.Rodrigues(rvec)[0]

        # Construct a homography from the object (plane) to the camera
        # image sensor
        H_cam_from_obj = plane_homog_from_KRt(K_cam, R, tvec)

        # Construct the inverse homography as well
        H_obj_from_cam = np.linalg.inv(H_cam_from_obj)

        proj_img_points_cam = all_proj_img_points_cam[i]
        
        # We can convert these camera sensor plane points to object frame by 
        # using the inverse of the homography we computed above
        proj_obj_points_2d = cv2.perspectiveTransform(proj_img_points_cam,
                                                      H_obj_from_cam)
                
                                                      
        #        print proj_img_points_cam
        #        print
        #        print "Homog from C"
        #        print
        #        print H_obj_from_cam
        #        
        # We can convert these 2D points into 3D points.
        # sets z=0
        proj_obj_points = np.dstack( (proj_obj_points_2d, 
                                      np.zeros((n_proj, 1, 1))))
        
        all_proj_obj_points.append(proj_obj_points.astype('float32'))

        all_proj_img_points_proj.append(proj_img_points_proj.astype('float32'))





    # Step 3:
    rval, K_proj, dists_proj, rvecs, tvecs = cv2.calibrateCamera(
        all_proj_obj_points, all_proj_img_points_proj,
        proj_size, flags=mono_flags)

    # Step 4:
    stereo_flags = (mono_flags |
                    cv2.CALIB_USE_INTRINSIC_GUESS | 
                    cv2.CALIB_FIX_PRINCIPAL_POINT | 
                    cv2.CALIB_FIX_ASPECT_RATIO | 
                    cv2.CALIB_FIX_INTRINSIC |
                    cv2.CALIB_FIX_FOCAL_LENGTH)

    # Estimate the extrinsic parameters of the projector
    rval, Kc2, dc2, Kp2, dp2, R, t, E, F = cv2.stereoCalibrate(
        all_proj_obj_points, all_proj_img_points_cam, all_proj_img_points_proj,
        cam_size, K_cam, dists_cam, K_proj, dists_proj)
    #    print
    #    print "K_proj:"
    #    print
    #    print K_proj
    #    print
    return K_cam, K_proj, R, t
               

######################################################################
# Camera intrinsics (extrinsics are trivial R=I, t=0)
# I totally just made these up. 
#
# We want to be sure that we can reconstruct these from our simulated
# data below.

cam_img_w = 640
cam_img_h = 480

K_cam = np.array([
    [ 651.0, 0.0, 322.0 ],
    [ 0.0, 649.0, 243.0 ],
    [ 0, 0, 1 ] ])

# need this below to call cv2.projectPoints
dists_cam = np.zeros(5)

######################################################################
# Projector intrinsics and extrinsics.
# These are also just made up.
# 
# Again, we want OpenCV to be able to estimate things very close to
# these.

proj_img_w = 800
proj_img_h = 600

# Made-up intrinsics
K_proj = np.array([
    [ 707.0, 0.0, 402.0 ],
    [ 0.0, 702.0, 293.0 ],
    [ 0, 0, 1] ])

# Extrinsics: 

# Constructing a rotation matrix from two vectors
# z axis of projector in camera frame
proj_z = normalize(np.array([0.3, 0.05, 1.0]))
up = normalize(np.array([0.0, 1.0, 0.0]))
proj_x = normalize(np.cross(up, proj_z))
proj_y = normalize(np.cross(proj_z, proj_x))

R_proj_from_cam = np.array([ proj_x, proj_y, proj_z ])

# Projector is to the left (negative x) of the camera in the camera
# frame
t_cam_from_proj = np.array([-0.5, 0.1, 0.02])

# To find the camera's position in the projector frame, do -R^T*t
t_proj_from_cam = -np.dot(R_proj_from_cam, t_cam_from_proj)

# Ainv maps image points in projector to directions in camera frame
Ainv = np.linalg.inv(np.dot(K_proj, R_proj_from_cam))


b = np.dot(K_proj, t_proj_from_cam)

# Omega is the projector's position in the camera frame
omega = -np.dot(Ainv, b)

# Sanity check
"""
print 'omega =', omega
print 'should be the same as t_cam_from_proj:', t_cam_from_proj
print
"""

######################################################################
# parameters of poster and printed grid

printed_grid_size = (6, 5)

poster_w = 1.6
poster_h = 1.2

printed_square_size = 0.05

# For simulation purposes only
poster_size = np.array([poster_w, poster_h])

# You probably actually care to measure this 
# Offset from BL grid square to the center of the poster
printed_origin = 2.0*printed_square_size

# Make a grid of 2D points on the poster
printed_obj_points_2d = make_grid_2d(printed_grid_size,
                                     printed_square_size,
                                     printed_origin)


n_printed = printed_obj_points_2d.shape[0]

# To convert 2D points to 3D object-frame points, I just tack on a zero
# sets z=0
printed_obj_points = np.dstack( (printed_obj_points_2d,
                                 np.zeros((n_printed,1,1))) )


# These are just for visualization
poster_corner_points = np.hstack(
    ( np.array( [
        poster_size * [0, 0],
        poster_size * [0, 1],
        poster_size * [1, 1],
        poster_size * [1, 0] ]),
      np.zeros((4,1)) ) )

poster_corner_points = poster_corner_points.reshape((4,1,3))

######################################################################
# Set up some stuff for the projector

proj_img_ctr = np.array([0.5*proj_img_w, 0.5*proj_img_h])

proj_square_size = 20

proj_grid_size = (8, 7)

proj_origin = proj_img_ctr-(np.array(proj_grid_size)*0.5-0.5)*proj_square_size

# You should know this going in -- the image (pixel) coordinates of the
# checkerboard corners in the projected image.
proj_img_points_proj = make_grid_2d(proj_grid_size, proj_square_size, proj_origin)

n_proj = proj_img_points_proj.shape[0]

# Homogeneous version of those coordinates (add 1 at the end)
# sets w=1
proj_img_points_homog = np.dstack( (proj_img_points_proj, np.ones((n_proj,1,1)) ) )

# Map projected image coordinates to camera-frame directions thru the
# Ainv matrix
proj_dirs = np.dot(proj_img_points_homog, Ainv.T)

# Reshape to OpenCV convention
proj_img_points_homog = proj_img_points_homog.reshape((-1, 1, 2))

proj_dirs = proj_dirs.reshape((-1, 1, 3))

######################################################################
# Synthesize us some data!

# Rotation vectors for successive poses of the posterboard
rvecs = np.array([
    [ 0.01, 0.2, 0.01 ],
    [ -0.2, 0.05, 0.08 ],
    [ 0.01, -0.3, 0.3 ],
])

# Translation vectors for the successive poses of the posterboard
tvecs = np.array([
    [ -0.8, -0.5, 2.0 ],
    [ -0.9, -0.4, 2.5 ],
    [ -0.4, -0.5, 2.8 ],
])

# Each rvec, tvec pair is a rigid transformation between the
# posterboard and the camera frame.  For instance, tvec[i] is the
# camera-frame position of the center of the posterboard in image i
# (and rvec[i] is the corresponding rotation of the board).

# Allocate a bunch of empty lists to store things that we will later
# pass to cv2.calibrateCamera and cv2.stereoCalibrate

all_printed_img_points = []
all_printed_obj_points = []

all_proj_img_points_cam = []
all_proj_img_points_proj = []
#all_proj_obj_points = []

all_images = []

# Loop over all poses (simulate the generation of all images):
for plane_rvec, plane_tvec in zip(rvecs, tvecs):

    # Rodrigues converts a rotation vector to a 3x3 rotation matrix
    # (or vice versa)
    plane_R = cv2.Rodrigues(np.array(plane_rvec))[0]


    # Apply the rigid transformation to convert the 2D printed object
    # points to 3D points in the camera frame.
    # (For simulation purposes only)
    p3d = plane_to_3d_from_Rt(plane_R, plane_tvec,  
                              printed_obj_points_2d)

    # Construct a homography from the object (plane) to the camera
    # image sensor
    H_cam_from_obj = plane_homog_from_KRt(K_cam, plane_R, plane_tvec)

    # Construct the inverse homography as well
    H_obj_from_cam = np.linalg.inv(H_cam_from_obj)


    
    # Extract the normal vector of the plane in the camera frame
    plane_n = plane_normal_from_R(plane_R)

    # There are a gazillion ways to get projected coordinates of 3D
    # points in the camera's sensor plane.

    # Way number 1: call cv2.projectPoints with the 3D camera-frame
    # points we have computed above, using no rotation or translation
    printed_img_points1 = cv2.projectPoints(p3d,
                                            np.zeros(3),
                                            np.zeros(3),
                                            K_cam, 
                                            dists_cam)[0]

    # Way number 2: call cv2.projectPoints with the object-frame points,
    # using the plane rotation vector and translation vector
    printed_img_points2 = cv2.projectPoints(printed_obj_points,
                                            plane_rvec, 
                                            plane_tvec, 
                                            K_cam, 
                                            dists_cam)[0]

    # Way number 3: use the homography from the object (plane) to the
    # camera.
    printed_img_points3 = cv2.perspectiveTransform(printed_obj_points_2d,
                                                    H_cam_from_obj)

    # Make sure these are all equivalent by computing the differences between them
    pdiff12 = np.abs(printed_img_points2 - printed_img_points1).max()
    pdiff13 = np.abs(printed_img_points3 - printed_img_points1).max()


    # Just for visualization, we show the outline of the poster
    outline_img_points = cv2.projectPoints(poster_corner_points,
                                           plane_rvec,
                                           plane_tvec,
                                           K_cam,
                                           dists_cam)[0]

    # Take the rays from the projector and intersect them with the
    # poster to get 3D camera-frame coordinates of the projected
    # corners
    proj_points_3d = project_to_plane_from_np(omega, proj_dirs,
                                              plane_n, plane_tvec)

    # We now can get camera sensor plane coordinates of the projected corners
    # by calling cv2.projectPoints
    proj_img_points_cam = cv2.projectPoints(proj_points_3d,
                                            np.zeros(3),
                                            np.zeros(3),
                                            K_cam,
                                            dists_cam)[0]

    # Append all of these points to the lists we allocated above.
    # Note they have to be in float32 format for OpenCV not to barf.
    all_printed_img_points.append( printed_img_points1.astype('float32') )
    all_printed_obj_points.append( printed_obj_points.astype('float32') )

    all_proj_img_points_cam.append( proj_img_points_cam.astype('float32') )
    all_proj_img_points_proj.append( proj_img_points_proj.astype('float32') )
    #all_proj_obj_points.append( proj_obj_points.astype('float32') )
    
    # Make an image to visualize some stuff
    img = np.zeros((cam_img_h, cam_img_w, 3), dtype='uint8')

    # Draw chessboard corners for printed and projected grids
    cv2.drawChessboardCorners(img, printed_grid_size, 
                              printed_img_points1.astype('float32'), True)

    cv2.drawChessboardCorners(img, proj_grid_size,
                              proj_img_points_cam.astype('float32'), True)

    # Draw the outline of the poster
    cv2.polylines(img, [outline_img_points.astype(int)], True, 
                  (255,255,255), 1, cv2.CV_AA)

    # Stash the image to display later
    all_images.append( img )


######################################################################
# Done with loop, now call some calibration stuff

K_cam2, K_proj2, R, t = proj_calib(printed_obj_points, proj_img_points_proj,
                                   all_printed_img_points,
                                   all_proj_img_points_cam,
                                   (cam_img_w, cam_img_h),
                                   (proj_img_w, proj_img_h))

print K_proj2.dtype

"""
print 'Camera calibration:'
print
print 'K_cam =\n', K_cam
print 'K_cam2 =\n', K_cam2
print 'err =\n', K_cam2-K_cam
print

print 'Projector calibration:'
print
print 'K_proj =\n', K_proj
print 'K_proj2 =\n', K_proj2
print 'err =\n', K_proj2-K_proj
print

print 'Stereo calibration:'
print 'R:\n', R
print 'R_proj_from_cam:\n', R_proj_from_cam
print 'err =\n', R-R_proj_from_cam
print
print 't: ', t.flatten()
print 't_proj_from_cam: ', t_proj_from_cam
print 'err = ', t.flatten()-t_proj_from_cam
print



win = 'Projector calibration'
cv2.namedWindow(win)

for img in all_images:
    cv2.imshow(win, img)
    while cv2.waitKey(5) < 0:
        pass
        
"""

