import numpy as np
import cv2
import argparse
import sys
from utils import load_stereo_coefficients, create_output
import matplotlib.pyplot as plt



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

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def generate_depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    # filteredImg = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)


    return filteredImg


if __name__ == '__main__':

    # is camera stream or video
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(2)
   

    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients('outputs/calib.txt')  # Get cams params

    if not cap_left.isOpened() and not cap_right.isOpened():  # If we can't get images from both sources, error
        print("Can't open the streams!")
        sys.exit(-1)

    while True:  # Loop until 'q' pressed or stream ends
        # Grab&retreive for sync images
        if not (cap_left.grab() and cap_right.grab()):
            print("No more frames")
            break

        _, leftFrame = cap_left.retrieve()
        _, rightFrame = cap_right.retrieve()
        height, width, channel = leftFrame.shape  # We will use the shape for remap

        # K1, D1, R1, P1,
        # 

        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)


        # We need grayscale for disparity map.
        gray_left = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)


        

        disparity_image = generate_depth_map(gray_left, gray_right) # Get the disparity map


        print('generating 3d point cloud...',)
        # f = 0.8*width#     i                     # guess for focal length
        # Q = np.float32([[1, 0, 0, -0.5*width],
        #                 [0,-1, 0,  0.5*height], # turn points 180 deg around x-axis,
        #                 [0, 0, 0,     -f], # so that y-axis looks up
        #                 [0, 0, 1,      0]])
        # points = cv2.reprojectImageTo3D(disparity_image, Q)
        # colors = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2RGB)
        # # mask = disparity_image > disparity_image.min()
        # out_points = points#[mask]
        # out_colors = colors#[mask]
        # out_fn = 'out.ply'
        # write_ply(out_fn, out_points, out_colors)
        # print('%s saved' % out_fn)



        # k  = cv2.waitKey(10000)

        # if k == 27:         # If escape was pressed exit
        #     cv2.destroyAllWindows()
        #     break

        # Show the images
        # cv2.imshow('left(R)', leftFrame)
        # cv2.imshow('right(R)', rightFrame)
        cv2.imshow('Disparity', disparity_image)
        cv2.waitKey(50)
        # print(disparity_image)
        # # plt.imshow(disparity_image, cmap='plasma')
        # plt.colorbar()
        # plt.show()

        # ''' Calc depth from B*F/disparity'''
        # B = 13
        # F = 1391.5
        # depth = B * F / disparity_image

        ''' Calc depth from reproject3D'''
        depth = cv2.reprojectImageTo3D(disparity_image, Q)



        print(depth)



    # Release the sources.
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()