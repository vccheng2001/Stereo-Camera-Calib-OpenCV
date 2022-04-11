# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy
import cv2
import cv2.aruco as aruco
import os
import pickle
import yaml 
import numpy as np
     
##################################################################
#       ESTIMATE CAMERA POSE (R/t) TO ARUCO BOARD
##################################################################

TYPE = "CHESSBOARD"
CAM_NUM = 1
VERSION = 13
np.set_printoptions(precision = 4, suppress = True)

if TYPE == "CHESSBOARD":
    d = {} 
    with open(f'CHESSBOARD/CAM{CAM_NUM}_calib_v2.yaml') as file:
        documents = yaml.full_load(file)

        for item, doc in documents.items():
            d[item] = np.array(doc) 

    camera_matrix = d['camera_matrix']
    dist_coeffs = d['dist_coeff']
    
else:
    camera_matrix, dist_coeffs = np.load('ARUCO/CAM{CAM_NUM}_calib_v{VERSION}.pckl', allow_pickle=True)



# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

# Creating a theoretical board we'll use to calculate marker positions
board = aruco.GridBoard_create(
    markersX=5,
    markersY=7,
    markerLength=0.04, # 4cm = 0.04 m
    markerSeparation=0.01,
    dictionary=ARUCO_DICT)

##################################################################
# Create vectors we'll be using for rotations and translations for postures
rvec, tvec = None, None
# find pose of Tcam->world in this video


videoFile = f'ARUCO_TWOCAMS_v{VERSION}/CAM{CAM_NUM}_vid_v{VERSION}.avi'
cam = cv2.VideoCapture(videoFile)

# ''' NOTE: ONLY IF USING MOBILE '''
# if CAM_NUM == "M":
#     videoFile = 'setup/CAMM/CAMM_aruco.mp4'
#     cam = cv2.VideoCapture(videoFile) # USE VIDEO
# else:
#     cam = cv2.VideoCapture(1)

i = 0
while(cam.isOpened()):
    i+=1
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()

    # cv2.imshow('q', QueryImg)
    # cv2.waitKey(0)
    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
    
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
  
        # Refine detected markers
        # Eliminates markers not part of our board, adds missing markers to the board
        corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
                image = gray,
                board = board,
                detectedCorners = corners,
                detectedIds = ids,
                rejectedCorners = rejectedImgPoints,
                cameraMatrix = camera_matrix,
                distCoeffs = dist_coeffs)   
        ###########################################################################
        # TODO: Add validation here to reject IDs/corners not part of a gridboard #
        ###########################################################################

        # print('corners', corners)

        # Outline all of the markers detected in our image
        QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

        # Require 15 markers before drawing axis
        if ids is not None and len(ids) > 15:
            # Estimate the posture of the gridboard, which is a construction of 3D space based on the 2D video 
            # object's origin in camera coordinate system
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, rvec, tvec)
            # print('rvec', rvec)
            # print('tvec', tvec)
            # rvec, tvec: board pose relative to the camera (Tworld_cam)
            if retval:
                # Draw the camera posture calculated from the gridboard
                QueryImg = aruco.drawAxis(QueryImg, camera_matrix, dist_coeffs, rvec, tvec, 0.3)
            
        # else:
        #     print('not enough')
        # print('final rvec', rvec)
        dst, jacobian = cv2.Rodrigues(rvec)

        
        print(dst)



        tcam_world = dst.T @ -tvec
        Rcam_world = dst.T 

        Tcam_world = np.matrix([[Rcam_world[0][0],Rcam_world[0][1],Rcam_world[0][2],tcam_world[0][0]],
                             [Rcam_world[1][0],Rcam_world[1][1],Rcam_world[1][2],tcam_world[1][0]],
                             [Rcam_world[2][0],Rcam_world[2][1],Rcam_world[2][2],tcam_world[2][0]],
                             [0.0, 0.0, 0.0, 1.0]
                ])



        # # Tcam_world = np.linalg.inv(Tworld_cam) # camera wrt board 
        # boardWrtCamera = np.array([Tworld_cam[0,3],Tworld_cam[1,3],Tworld_cam[2,3]])

        # print(boardWrtCamera)

        data = {f'Tcam{CAM_NUM}_world': np.asarray(Tcam_world).tolist()}
        with open(f"ARUCO_TWOCAMS_v{VERSION}/CAM{CAM_NUM}_T_v{VERSION}.yaml", "w") as f:
            yaml.dump(data, f)

        print('T cam world', Tcam_world)
        # Display our image
        cv2.imshow('QueryImage', QueryImg)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()