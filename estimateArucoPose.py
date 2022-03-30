# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.

import numpy as np
import cv2
import cv2.aruco as aruco
import sys
import os
import pickle
import yaml 

TYPE = "CHESSBOARD"
CAM_NUM = 2
VERSION = 2
if TYPE == "CHESSBOARD":
    d = {} 
    with open(f'CHESSBOARD/CAM{CAM_NUM}_calib_v{VERSION}.yaml') as file:
        documents = yaml.full_load(file)

        for item, doc in documents.items():
            d[item] = np.array(doc) 

    camera_matrix = d['camera_matrix']
    dist_coeffs = d['dist_coeff']
    
else:
    camera_matrix, dist_coeffs = np.load('ARUCO/CAM{CAM_NUM}_calib_v{VERSION}.pckl', allow_pickle=True)



# Check for camera calibration data

def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    # img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

# Creating a theoretical board we'll use to calculate marker positions
board = aruco.GridBoard_create(
    markersX=5,
    markersY=7,
    markerLength=0.04,
    markerSeparation=0.01,
    dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rotation_vectors, translation_vectors = None, None
axis = np.float32([[-.5,-.5,0], [-.5,.5,0], [.5,.5,0], [.5,-.5,0],
                   [-.5,-.5,1],[-.5,.5,1],[.5,.5,1],[.5,-.5,1] ])

# Make output image fullscreen
cv2.namedWindow('ProjectImage',cv2.WINDOW_NORMAL)

cam = cv2.VideoCapture(1)
# cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
# cam.set(3, 1280)
# cam.set(4, 720)
while(cam.isOpened()):
    # Capturing each frame of our video stream
    ret, ProjectImage = cam.read()
    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(ProjectImage, cv2.COLOR_BGR2GRAY)
        
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

        # Outline all of the markers detected in our image
        # Uncomment below to show ids as well
        # ProjectImage = aruco.drawDetectedMarkers(ProjectImage, corners, ids, borderColor=(0, 0, 255))
        ProjectImage = aruco.drawDetectedMarkers(ProjectImage, corners, borderColor=(0, 0, 255))

        # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        # Require at least 1 marker before drawing axis
        if ids is not None and len(ids) > 0:
            # Estimate the posture per each Aruco marker
            rotation_vectors, translation_vectors, _objPoints = aruco.estimatePoseSingleMarkers(corners, 1, camera_matrix, dist_coeffs)
            
            for rvec, tvec in zip(rotation_vectors, translation_vectors):
                if len(sys.argv) == 2 and sys.argv[1] == 'cube':
                    try:
                        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
                        ProjectImage = drawCube(ProjectImage, corners, imgpts)
                    except:
                        continue
                else:    
                    ProjectImage = aruco.drawAxis(ProjectImage, camera_matrix, dist_coeffs, rvec, tvec, 1)

        cv2.imshow('ProjectImage', ProjectImage)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()