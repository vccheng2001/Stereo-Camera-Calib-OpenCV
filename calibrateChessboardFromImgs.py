import numpy as np
import cv2 as cv
import glob
import yaml

''' Camera calibration 
Procedure: Take 10+ photos of chessboard from various viewpoints
to calibrate camera. 
Returns intrinsic matrix and calculates reprojection errors
'''

print('****** CALIBRATING CHESSBOARD FROM IMAGES *********\n')


# ``columns`` and ``rows`` should be the number of inside corners in the
# chessboard's columns and rows. ``show`` determines whether the frames
# are shown while the cameras search for a chessboard.

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
CHECKERBOARD = (6,9) # cols, rows 
print('init check')
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
files = []

TYPE="CHESSBOARD"
CAM_NUM = 2
VERSION = 2
images = glob.glob(f'CHESSBOARD/CAM{CAM_NUM}_imgs_v{VERSION}/*.jpg')
for fname in images:
    print(f'Processing image {fname}', end=' ')
    files.append(fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print('Success!')
        objpoints.append(objp)
        # iterative process to refine corner location
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv.imshow('img', img) # show
        # cv.waitKey(500)
    else:
        print("Failed!")
cv.destroyAllWindows()


# pass the 3D points in world coordinates and their 2D locations in all images 
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

data = {'camera_matrix': np.asarray(cameraMatrix).tolist(), 'dist_coeff': np.asarray(distCoeffs).tolist()}

with open(f"CHESSBOARD/CAM{CAM_NUM}_calib_v{VERSION}.yaml", "w") as f:
    yaml.dump(data, f)

print(f"Camera matrix : {cameraMatrix}\n")
print(f"dist : {distCoeffs}\n")
print(f"rvecs : {rvecs}\n")
print(f"tvecs : {tvecs}\n")


''' Print reprojection error for each file'''
mean_error = 0
for i in range(len(objpoints)):
    print(f'File {files[i]}')

    # imgpoints2: we first transform the object point to image point using cv2.projectPoints()
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    print(f'error {i}: {error}')
    mean_error += error
print( "\nTotal error: {}".format(mean_error/len(objpoints)) )

with open(f"CHESSBOARD/CAM{CAM_NUM}_error_v{VERSION}.txt", 'w') as f:
    f.write(f'mean_error: {mean_error/len(objpoints)}')

print('****** DONE CALIBRATING ARUCO FROM IMAGES *********\n')
