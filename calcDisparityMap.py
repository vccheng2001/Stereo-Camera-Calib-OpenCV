import numpy as np
import cv2
import argparse
import sys
from utils import *
import matplotlib.pyplot as plt
# np.set_printoptions(suppress=True)


'''

Visualize disparity/depth map from two stereo cameras in real time.
This script reads images from two video streams, then undistorts and rectifies each
frame using stereo coefficients (see stereoCalibrateAndRectify.py). 

After the transformation is given to remap, we’ll get the rectified images. 
Then, we can call gen_depth_map()on left and right images to perform Stereo SGBM
matching with WLS filter to obtain a smooth disparity map. 

'''


def gen_depth_map(imgL, imgR):

    # SGBM Parameters 
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=360,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
    # Weighted least squares filter to fill sparse (unpopulated) areas of the disparity map
        # by aligning the images edges and propagating disparity values from high- to low-confidence regions
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    # Get depth information/disparity map using SGBM
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    # filteredImg = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)

    return filteredImg 



def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')




def main(args):
    # read in camera streams 
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(2)

    # load stereo coefficients (see stereoCalibrateAndRectify.py)
    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.out_file)  # Get cams params

    if not cap_left.isOpened() and not cap_right.isOpened():  # If we can't get images from both sources, error
        print("Can't open the streams!")
        sys.exit(-1)

    while True:  # Loop until 'q' pressed or stream ends
        if not (cap_left.grab() and cap_right.grab()):
            print("No frames to read.")
            break

        _, leftFrame = cap_left.retrieve()
        _, rightFrame = cap_right.retrieve()

        h, w, c = leftFrame.shape
        

        # undistort and rectify map, then remap 
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w,h), cv2.CV_32FC1)
        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w,h), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        # stacked = np.hstack((gray_left, gray_right))
        # cv2.imshow('stacked', stacked)
        # cv2.waitKey(0)
        # exit(-1)
        
        # Optional: downsample image 
        # gray_left = downsample_image(gray_left, 0.5)
        # gray_right = downsample_image(gray_right, 0.5)

        disparity_image = gen_depth_map(gray_left, gray_right)


        # Get Baseline, Focal length from Q matrix     
        B =  -1/Q[3][2]
        F = Q[2][3]
        depth =  (B * F / disparity_image)


        h, w = gray_left.shape

        # Select points to label depth 
        centers = []
        centers.append([h // 2, w // 4]) 
        centers.append([h // 2, w // 2]) # find center of image 
        centers.append([h - (h // 4), w - (w// 2)])
        centers.append([h - (h // 4), w - (w// 4)])

        colored_disparity = disparity_image 

        # Visualize depth map, label depth at select points 
        for i, j in centers:
            try:
                colored_disparity = cv2.cvtColor(colored_disparity,cv2.COLOR_GRAY2RGB)
            except:
                pass           
            try:
                cv2.circle(colored_disparity, (j,i), 15, (255,0,0), -1)
                cv2.putText(colored_disparity, f'{str(round(depth[i][j], 3))}m', (j+10,i+10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255,0,0), 2)
            except:
                pass

        cv2.imshow('Disparity', colored_disparity)
        cv2.waitKey(1)
   

        # Calc depth from reproject3D
        # depth = cv2.reprojectImageTo3D(disparity_image, Q)


    # Release the sources.
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize disparity map")
    parser.add_argument("--out_file", default='stereo_coeffs.txt', type=str, help="file to store stereo coefficients")

    args = parser.parse_args()
    main(args)
    