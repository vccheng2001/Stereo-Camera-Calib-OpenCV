import numpy as np
import cv2
import argparse
import sys
from utils import load_stereo_coefficients, create_output
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)



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

VERSION = 19

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

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

    return filteredImg 



def generate_depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    # SGBM Parameters -----------------
    window_size = 3                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
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
    sigma = 1.5
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    print('computing disparity...')
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    # cv2.imshow('Disparity Map', filteredImg)
    # filteredImg = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)


    return filteredImg


if __name__ == '__main__':

    # is camera stream or video
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(2)
   

    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(f'outputs/calib_v{VERSION}.txt')  # Get cams params

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

        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)


        # # We need grayscale for disparity map.
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)


        # imgL = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR)
        # imgR = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR)


        stacked = np.hstack((gray_left, gray_right))
        # cv2.imshow('stacked', stacked)
        # cv2.waitKey(0)
        # exit(-1)
        # 
        

        disparity_image = gen_depth_map(gray_left, gray_right)


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
        
        # print(disparity_image)
        # # plt.imshow(disparity_image, cmap='plasma')
        # plt.colorbar()
        # plt.show()

        # ''' Calc depth from B*F/disparity'''

        
        B = -1/Q[3][2]
        F = Q[2][3]

        # print('B', B, 'F', F, 'BF', B*F)
        depth = B * F / disparity_image


        ''' Label depths for select points '''
        # center (width/2, height/2)
        centers = []
        centers.append([disparity_image.shape[1] // 2, disparity_image.shape[0] // 4])
        centers.append([disparity_image.shape[1] // 2, disparity_image.shape[0] // 2])
        centers.append([disparity_image.shape[1] // 4, disparity_image.shape[0] // 2])


        for center in centers:

            center_depth = depth[center[0]][center[1]]
            cv2.circle(disparity_image, center, 15, (0, 255, 0), -1)
            cv2.putText(disparity_image, str(center_depth), center, cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Disparity', disparity_image)
        cv2.waitKey(1)
   
        


        # print('deepp', depth[1080//2-1][1920//2-1])

        # print(K1)
        # print(K2)
        # print('Q', Q)
        # exit(-1)
        ''' Calc depth from reproject3D'''
        # depth = cv2.reprojectImageTo3D(disparity_image, Q)

        hist, bins = np.histogram(depth, bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,100 ])
        print(hist)
        print(bins)
        




        # print(depth)



    # Release the sources.
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()