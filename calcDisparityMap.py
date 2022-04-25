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

VERSION = 23

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
        numDisparities=480,  # max_disp has to be dividable by 16 f. E. HH 192, 256
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

        # cv2.imshow('leftFrame', leftFrame)
        # cv2.waitKey(0)
        # exit(-1)

        
        height, width, channel = leftFrame.shape  # We will use the shape for remap

        # K1, D1, R1, P1,
        # 


        '''
        initUndistortRectifyMap function both undistorts and rectifies the images. 
        For the left camera, we use K1(camera matrix) and D1(distortion matrix) to undistort 
        and R1(left to right rotation) and P1(left to right projection matrix) to rectify.

        After the transformation is given to remap, we’ll get the rectified images. 


        What remap() does do is, for every pixel in the destination image,
        lookup where it comes from in the source image, and then assigns an interpolated value.

        At pixel (0, 0) in the new destination image, I look at map_x and map_y which tell me the location of the corresponding pixel
        in the source image, and then I can assign an interpolated value at (0, 0) in the
        destination image by looking at near values in the source. 


        '''

        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # left_rectified =cv2.flip(left_rectified, 0)
        # right_rectified =cv2.flip(right_rectified, 0)
        # cv2.imshow('left rec', left_rectified)
        # cv2.waitKey(0)
        # exit(-1)
        # # We need grayscale for disparity map.
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        # h, w = gray_left.shape
        # i, j = [h // 2, w // 2]
        # print('orig center', i, j)
        # ii = int(leftMapY[i][j])
        # jj = int(leftMapX[i][j])

        

        # # print('leftMapX', leftMapX)
        # cv2.circle(left_rectified, (ii,jj), 15, (255,255,0), -1)
        # cv2.imshow('left rect',  left_rectified)
        # cv2.waitKey(0)
        # exit(-1)


        # imgL = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR)
        # imgR = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR)


        # stacked = np.hstack((gray_left, gray_right))
        # cv2.imshow('stacked', stacked)
        # cv2.waitKey(0)
        # exit(-1)
        
        

        disparity_image = gen_depth_map(gray_left, gray_right)


        # print('generating 3d point cloud...',)
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
        
       
        # # plt.imshow(disparity_image, cmap='plasma')
        # plt.colorbar()
        # plt.show()

        # ''' Calc depth from B*F/disparity'''

        
        B =  -1/Q[3][2]
        F = Q[2][3]

        print('B', B, 'F', F, 'BF', B*F)
        depth =  (B * F / disparity_image)#/ 100 # meteres


        h, w = disparity_image.shape
        ''' Label depths for select points '''

        centers = []
        centers.append([h // 2, w // 4]) 
        centers.append([h // 2, w // 2]) # find center of image 
        centers.append([h - (h // 4), w - (w// 2)])
        centers.append([h - (h // 4), w - (w// 4)])

        colored_disparity = disparity_image 

        for i, j in centers:

            depth_src = depth[i][j] # corresponds to src image 

            # there'll be an offset on disparity image 
            try:
                colored_disparity = cv2.cvtColor(colored_disparity,cv2.COLOR_GRAY2RGB)
            except:
                pass
            cv2.circle(colored_disparity, (j,i), 15, (255,0,0), -1)
            cv2.putText(colored_disparity, f'{str(round(depth_src,3))}m', (j+10,i+10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,0,0), 2)

        cv2.imshow('Disparity', colored_disparity)
        cv2.waitKey(1)
   
        


        # print('deepp', depth[1080//2-1][1920//2-1])

        # print(K1)
        # print(K2)
        # print('Q', Q)
        # exit(-1)
        ''' Calc depth from reproject3D'''
        # depth = cv2.reprojectImageTo3D(disparity_image, Q)

        hist, bins = np.histogram(depth, bins = [0, 50, 100, 200,  500, 1000, 2500, 5000])
        print(hist)
        print(bins)
        




        # print(depth)



    # Release the sources.
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()