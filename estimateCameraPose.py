
import numpy
import cv2
import cv2.aruco as aruco
import os
import pickle
import yaml 
import numpy as np
np.set_printoptions(precision = 4, suppress = True)
import os
     
'''
Estimate camera pose (rotation, translation) with respect to 
the plane of aruco markers. 

Requires calibrated camera (see calib_left.yaml or calib_right.yaml)

Saves resulting Tcam{left}_world as a yaml file 
Once you get both Tcam{left}_world and Tcam{right}_world, 
you can run calcRelativePose.py to get the relative pose between the
two cameras. 
'''

def main(args): 
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


    cam = cv2.VideoCapture(0)

    i = 0
    while(cam.isOpened()):
        i+=1
        # Capturing each frame of our video stream
        ret, QueryImg = cam.read()
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
        
            # Outline all of the markers detected in our image
            QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

            # Require 15 markers before drawing axis
            if ids is not None and len(ids) > 15:
                # Estimate the posture of the gridboard, which is a construction of 3D space based on the 2D video 
                # object's origin in camera coordinate system
                retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, rvec, tvec)
                # rvec, tvec: board pose relative to the camera (Tworld_cam)
                if retval:
                    # Draw the camera posture calculated from the gridboard
                    QueryImg = aruco.drawAxis(QueryImg, camera_matrix, dist_coeffs, rvec, tvec, 0.3)
                
            # convert rotation vec to matrix
            dst, jacobian = cv2.Rodrigues(rvec)

            # Find trans, rotation of camera wrt plane of Aruco marketsr 
            tcam_world = dst.T @ -tvec
            Rcam_world = dst.T 

            # Get homog transform: camera wrt wrold 
            Tcam_world = np.matrix([[Rcam_world[0][0],Rcam_world[0][1],Rcam_world[0][2],tcam_world[0][0]],
                                    [Rcam_world[1][0],Rcam_world[1][1],Rcam_world[1][2],tcam_world[1][0]],
                                    [Rcam_world[2][0],Rcam_world[2][1],Rcam_world[2][2],tcam_world[2][0]],
                                    [0.0, 0.0, 0.0, 1.0]
                                    ])



            data = {f'Tcam{args.type}_world': np.asarray(Tcam_world).tolist()}
            with open(f"Tcam{args.type}_world.yaml", "w") as f:
                yaml.dump(data, f)


            # Display our image
            cv2.imshow('QueryImage', QueryImg)

        # Exit at the end of the video on the 'q' keypress
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimates T_cam_to_world")
    parser.add_argument("--type", type=str, default='left', help="Specify left or right camera")
    args = parser.parse_args()
    main(args)
    