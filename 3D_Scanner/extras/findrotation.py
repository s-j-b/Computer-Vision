import cv2
import numpy as np
from calibrate import calibData
import scanner

# This program uses a calibrated projector-camera setup and a turntable to obtain a 3D point cloud
# for a scanned object


def do_kdtree(combined_x_y_arrays,points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return indexes



def getRotation(board0,board20,library0,library20,K_cam,K_proj,h,R_t,oP):
    
    patternSize = (9,6)
    
    board0 = cv2.resize(board0,(w,h))
    board20 = cv2.resize(board20,(w,h))

    ret, sensorPts0 = cv2.findChessboardCorners(board0,
                                                patternSize,
                                                flags = cv2.CALIB_CB_ADAPTIVE_THRESH)

    ret, sensorPts20 = cv2.findChessboardCorners(board20,
                                                 patternSize,
                                                 flags = cv2.CALIB_CB_ADAPTIVE_THRESH)

    x0,y0,z0 = np.hsplit(library0,3)
    x20,y20,z20 = np.hsplit(library0,3)
    
    axes0 = np.hstack((x0,y0))
    axes20 = np.hstack((x20,y20))
    
    inds0 = do_kdtree(axes0,sensorPts0)
    inds20 = do_kdtree(axes20,sensorPts20)

    #search sensorpts0,20, in libraries, get an x for each
    



    #NOW WE HAVE 54 SENSOR PLANE POINTS, WITH CORRESPONDING X VALUES, OF THE FORM
    #IN LIST WE CALL points[0,1]
    # [U,V,X ]
    
    
    #Get A_inv matrix to find w1,w2
    K_proj_inv = np.linalg.inv(K_proj)
    A_inv = np.dot(R_t,K_proj_inv)
    K_cam_inv = np.linalg.inv(K_cam)
    
    pts0 = []
    pts20 = []
    for i in range(2):
        for j in range(54):
            
            q = points[i][j]
            u = q[2]

            #top and bottom points of green line
            q_prime1 = np.array([u,0,1])
            q_prime2 = np.array([u,h,1])

            w1 = np.dot(A_inv,q_prime1)
            w2 = np.dot(A_inv,q_prime2)

            #get the normal of the plane defined by the two vectors
            w_cross = np.cross(w1,w2)
            normal = w_cross/np.linalg.norm(w_cross)
            
            q[2] = 1
            
            P = np.dot(K_cam_inv,q)                                 #unscaled ray
            
            alpha = np.dot(normal,oP)/np.dot(normal,P)     #based on definition of where a ray and plane intersect
            
            P = P*alpha
            if i==0:
                pts0.append(P)
            if i ==1:
                pts20.append(P)
























def main():
    
    # R and t are the projector Extrinsics
    #k_cam and k_proj are the cam and projector intrinsics
    
    K_cam, K_proj, R, t = calibData
    
    
    
    
    
    
    
    w, h, lines0, x0s, cam, = scanner.scan()
    
    print'Turn on light, then press a key'
    
    #wait for keypress
    #save image
    cv2.waitKey()
    ret,board0 = cam.read()
    
    print'rotate 20 degrees, then press a key'
    
    
    #wait for keypress
    #save image
    cv2.waitKey()
    ret,board20 = cam.read()
    
    
    print 'Turn off light and press any key to continue'
    #wait for keypress
    cv2.waitKey()
    z, a, lines20, x20s, c = scanner.scan()
    
    
    
    
    library0 = getpts(lines0)
    library20 = getpts(lines20)
    
    
    
    
    
    #the optical center of the projector
    R_t = np.transpose(R)
    oP = -np.dot(R_t,t)
    
    Hrot = getRotation(board0,board20,library0,library20,K_cam,K_proj,h,R_t,oP)
    
    
    
    
    
    
    
    
    
    depth_points = np.array(depth_points)


def getpts(plist, xvals):
    for i in range(len(plist)):
        plist[i][:][2] = xvals[i]
    return pList




main()
    





