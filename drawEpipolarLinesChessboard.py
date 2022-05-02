

import cv2
import numpy as np
from utils import load_stereo_coefficients
np.set_printoptions(suppress=True)
import os
import argparse


''' Draws epipolar lines by finding object corners from chessboard pattern
in left/right images'''

# find object corners from chessboard pattern  and create a correlation with image corners
def getCorners(images, chessboard_size, show=True):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)*3.88 # multiply by 3.88 for large chessboard squares

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for image in images:
        frame = cv2.imread(image)
        # height, width, channels = frame.shape # get image parameters
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)   # Find the chess board corners
        if ret:                                                                         # if corners were found
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)    # refine corners
            imgpoints.append(corners2)                                                  # add to corner array

            if show:
                # Draw and display the corners
                frame = cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
                cv2.imshow('frame', frame)
                cv2.waitKey(100)

    cv2.destroyAllWindows()             # close open windows
    return objpoints, imgpoints, gray.shape[::-1]

# perform undistortion on provided image
def undistort(image, mtx, dist):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    print(image)
    # image = os.path.splitext(image)[0]
    h, w = img.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return dst

# draw the provided points on the image
def drawPoints(img, pts, colors):
    for pt, color in zip(pts, colors):
        
        center = tuple(map(int, pt[0]))
        cv2.circle(img, center, 15, int(color), -1)

# draw the provided lines on the image
def drawLines(img, lines, colors):
    _, c, _ = img.shape
    for r, color in zip(lines, colors):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img, (x0, y0), (x1, y1), int(color), 4)


def main(args):

    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.stereo_coeffs)  # Get cams params
    
    chessboard_size = (6,9)
    # undistort our chosen images using the left and right camera and distortion matricies
    imgL = undistort(args.img_left, K2, D2)
    imgR = undistort(args.img_right, K1,D1)
   
    imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

    # use get corners to get the new image locations of the checcboard corners (undistort will have moved them a little)
    _, imgpointsL, _ = getCorners([args.img_left], chessboard_size, show=False)
    _, imgpointsR, _ = getCorners([args.img_right], chessboard_size, show=False)

    # get 3 image points of interest from each image and draw them
    ptsL = np.asarray([imgpointsL[0][0], imgpointsL[0][10], imgpointsL[0][20]])
    ptsR = np.asarray([imgpointsR[0][5], imgpointsR[0][15], imgpointsR[0][25]])

    colors = (list(np.random.choice(range(256), size=50)))  
    drawPoints(imgL, ptsL, colors[3:6])
    drawPoints(imgR, ptsR, colors[0:3])

    # find epilines corresponding to points in right image and draw them on the left image
    epilinesR = cv2.computeCorrespondEpilines(ptsR.reshape(-1, 1, 2), 2, F.T)
    epilinesR = epilinesR.reshape(-1, 3)
    drawLines(imgL, epilinesR, colors[0:3])

    # find epilines corresponding to points in left image and draw them on the right image
    epilinesL = cv2.computeCorrespondEpilines(ptsL.reshape(-1, 1, 2), 1, F.T)
    epilinesL = epilinesL.reshape(-1, 3)
    drawLines(imgR, epilinesL, colors[3:6])

    stacked = np.hstack((imgL, imgR))
    cv2.imshow('Stacked', stacked)
    cv2.waitKey(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Draw Epipolar Lines with Chessboard")
    parser.add_argument("--img_left", type=str,default="capture_left/10.jpg", help="image captured by left camera")
    parser.add_argument("--img_right", type=str,default="capture_right/10.jpg",  help="image captured by right camera")
    parser.add_argument("--stereo_coeffs", type=str,default="stereo_coeffs.txt",  help="stereo coefficents file")

    args = parser.parse_args()
    main(args)
