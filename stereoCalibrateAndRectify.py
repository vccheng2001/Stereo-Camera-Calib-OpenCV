import numpy as np
import cv2
import glob
import argparse
import sys
from utils import yaml_load
from utils import save_stereo_coefficients


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_size = None
np.set_printoptions(suppress=True)



def stereo_calibrate(args):
    """ Stereo calibration and rectification """
    objp, leftp, rightp = load_image_points()

    
    K1 = np.array(yaml_load(args.calib_right_file, 'camera_matrix'))
    D1 = np.array(yaml_load(args.calib_right_file, 'dist_coeff'))
    K2 = np.array(yaml_load(args.calib_left_file, 'camera_matrix'))
    D2 = np.array(yaml_load(args.calib_left_file, 'dist_coeff'))

    flag = 0
    flag |= cv2.CALIB_USE_INTRINSIC_GUESS 
    flag |= cv2.CALIB_FIX_FOCAL_LENGTH 
    flag |= cv2.CALIB_FIX_PRINCIPAL_POINT

    # flag |= cv2.CALIB_FIX_INTRINSIC
    # flag |= cv2.CALIB_USE_INTRINSIC_GUESS
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objp, rightp, leftp, K1, D1, K2, D2, image_size)
    print("Stereo calibration rms: ", ret)

    # https://amroamroamro.github.io/mexopencv/matlab/cv.stereoRectify.html
    # Computes rectification transforms for each head of a calibrated stereo camera

    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
    K1,roi_right = cv2.getOptimalNewCameraMatrix(K1, D1,(args.width,args.height),0,(args.width,args.height))
    K2,roi_left = cv2.getOptimalNewCameraMatrix(K2, D2,(args.width,args.height),0,(args.width,args.height))



    print("K1", K1)
    print("K2", K2)
    print("Q", Q)
    save_stereo_coefficients(args.out_file, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q)


def load_image_points():
    global image_size
    width = 6
    height = 9
    square_size = 0.025
    pattern_size = (width, height)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size  # Create real world coords

    objpoints = []  # 3d point in real world space
    left_imgpoints = []  # 2d points in image plane.
    right_imgpoints = []  # 2d points in image plane.
    
    
    left_images = glob.glob(f'{args.left_dir}/*.jpg')
    right_images = glob.glob(f'{args.right_dir}/*.jpg')

    print(len(left_images), len(right_images))

    # images should be paired
    left_images.sort()
    right_images.sort()

    # Pairs should be same size. Otherwise we have sync problem.
    if len(left_images) != len(right_images):
        print("Numbers of left and right images are not equal. They should be pairs.")
        print("Left images count: ", len(left_images))
        print("Right images count: ", len(right_images))
        sys.exit(-1)

    pair_images = zip(left_images, right_images)  # Pair the images for single loop handling

    # loop through pairs of images 
    for left_im, right_im in pair_images:

        # right object points
        right = cv2.imread(right_im)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size,
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        # left object points
        left = cv2.imread(left_im)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        if ret_left and ret_right:  
            print('OK')
            # Object points
            objpoints.append(objp)
            # Right points
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (5, 5), (-1, -1), criteria)
            right_imgpoints.append(corners2_right)
            # Left points
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (5, 5), (-1, -1), criteria)
            left_imgpoints.append(corners2_left)
        else:
            print("Chessboard couldn't be detected. Image pair: ", left_im, " and ", right_im)
            continue

    image_size = gray_right.shape  # If you have no acceptable pair, you may have an error here.
    return [objpoints, left_imgpoints, right_imgpoints]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate and Rectify")
    parser.add_argument("--calib_right_file", default='calib_right.yaml', type=str, help="calib file for right camera")
    parser.add_argument("--calib_left_file", default='calib_left.yaml', type=str, help="calib file for right camera")
    parser.add_argument("--out_file", default='stereo_coeffs.txt', type=str, help="file to store stereo coefficients")
    parser.add_argument("--left_dir", default='capture_left', type=str, help="left images dir")
    parser.add_argument("--right_dir", default='capture_right', type=str, help="right images dir")
    parser.add_argument("--width",  default=1920,  help="resolution width")
    parser.add_argument("--height", default=1080,  help="resolution height")

    
    args = parser.parse_args()
    stereo_calibrate(args)
