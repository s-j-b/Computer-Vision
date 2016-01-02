######################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@                        ___________               @#
#@                       |____ |  _  \              @#
#@   ___  ___ __ _ _ __      / / | | |_ __  _   _   @#
#@  / __|/ __/ _` | '_ \     \ \ | | | '_ \| | | |  @#
#@  \__ \ (_| (_| | | | |.___/ / |/ /| |_) | |_| |  @#
#@  |___/\___\__,_|_| |_|\____/|___(_) .__/ \__, |  @#
#@                                   | |     __/ |  @#
#@                                   |_|    |___/   @#
#@                                                  @#
#@@@@@@@@ by Simon J. Bloch and Gibson Cook @@@@@@@@@#
######################################################

import cv2
import numpy as np

#Run calibrate.py and get data
from calibrate import calibData

#Import scan function from scan
from scanner import scan

def main():
    
    #Parse out the calibration data
    K_cam, K_proj, R, t, dists = calibData
    
    #Call scan function and get output data
    # - q_lines_list: lines data
    # - xvals: horizontal coordinates of the green line
    w, h, q_lines_list, xvals, win = scan(K_cam, dists)
    
    #Get R transpose
    R_t = np.transpose(R)
    
    #Get omega prime, the optical center of the projector
    noP = np.dot(R_t,t)
    oP = noP*(-1)
    
    #get every 3D point
    depth_points = get_one_view(K_proj,K_cam,h,q_lines_list,xvals,R,R_t, oP)
    depth_points = np.array(depth_points)
    
    #saving as a .npz file that we can open with PointCloudApp.py
    np.savez("finalTest",depth_points)

def get_one_view(K_proj,K_cam,h,q_lines_list,xvals,R,R_t,oP):
    
    #empty list of 3D points
    depth_points = []
    
    #iterate through x-values
    j = 0
    
    #for each line, get all 3D points
    for line in q_lines_list:

        u = xvals[j]
        j+=1
        
        #top and bottom points of green line
        q_prime1 = np.array([u,0,1])
        q_prime2 = np.array([u,h,1])            
        
        #inverse of intrinsics
        K_proj_inv = np.linalg.inv(K_proj)
        K_cam_inv = np.linalg.inv(K_cam)
        
        #Get A_inv matrix to find w1,w2
        A_inv = np.dot(R_t,K_proj_inv)
        
        #vectors along top and bottom of projected plane
        w1 = np.dot(A_inv,q_prime1)         
        w2 = np.dot(A_inv,q_prime2)            
        
        #get the normal of the plane defined by the two vectors
        w_cross = np.cross(w1,w2)
        normal = w_cross/np.linalg.norm(w_cross)
        
        
        #finding the depth for each point
        for q in line:

            #unscaled ray
            P = np.dot(K_cam_inv,q)
            
            #Based on definition of where a ray and plane intersect,
            # we find alpha, the scalar by which we will multiply our points
            # to establish their 3D address
            alphaNum = np.dot(normal,oP)
            alphaDenom = np.dot(normal,P)
            alpha = alphaNum*1.0/alphaDenom
            
            x = P[1]
            y = P[0]
            
            #Scale
            P *= alpha
            
            #Remove background
            if alpha < 1.5:
                depth_points.append(P)

    return depth_points

main()
    





