''' 

Calculates disparity map from two images  

'''
import cv2
import numpy as np
from utils import yaml_load
import matplotlib.pyplot as plt


#Function that Downsamples image x number (reduce_factor) of times. 
def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image


#=========================================================
# Stereo 3D reconstruction 
#=========================================================
VERSION = 13
SET = 1

calib1_file = 'CHESSBOARD/CAM1_calib_v2.yaml'
calib2_file = 'CHESSBOARD/CAM2_calib_v2.yaml'

K1 = np.array(yaml_load(calib1_file, 'camera_matrix'))
dist1 = np.array(yaml_load(calib1_file, 'dist_coeff'))

K2 = np.array(yaml_load(calib2_file, 'camera_matrix'))
dist2 = np.array(yaml_load(calib2_file, 'dist_coeff'))

#Specify image paths
img1_path = f'3D_v{VERSION}/set{SET}/cam1.jpg'
img2_path = f'3D_v{VERSION}/set{SET}/cam2.jpg'
#Load pictures
img1 = cv2.imread(img1_path,0)
img2 = cv2.imread(img2_path,0)


# Initialize the stereo block matching object 
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)

# Compute the disparity image
disparity = stereo.compute(img1, img2)

# Normalize the image for representation
min = disparity.min()
max = disparity.max()
disparity = np.uint8(6400 * (disparity - min) / (max - min))

# Display the result
cv2.imshow('disparity', np.hstack((img1, img2, disparity)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# print(f'K1:{K1}, dist1:{dist1}')
# print(f'K2:{K2}, dist2:{dist2}')

# #Specify image paths
# img1_path = f'3D_v{VERSION}/set{SET}/cam1.jpg'
# img2_path = f'3D_v{VERSION}/set{SET}/cam2.jpg'
# #Load pictures
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)

# #Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size 
# h,w = img2.shape[:2]

# #Get optimal camera matrix for better undistortion 
# K1, roi1 = cv2.getOptimalNewCameraMatrix(K1,dist1,(w,h),1,(w,h))
# K2, roi2 = cv2.getOptimalNewCameraMatrix(K2,dist2,(w,h),1,(w,h))

# #Undistort images
# img1 = cv2.undistort(img1, K1, dist1, None, K1)
# img2 = cv2.undistort(img2, K2, dist2, None, K2)

# # cv2.imshow('img1', img1)
# # cv2.waitKey(0)

# # cv2.imshow('img2', img2)
# # cv2.waitKey(0)


# #Downsample each image 3 times (because they're too big)
# img1 = downsample_image(img1,3)
# img2 = downsample_image(img2,3)



# #Set disparity parameters
# #Note: disparity range is tuned according to specific parameters obtained through trial and error. 
# win_size = 5
# min_disp = -1
# max_disp = 63 #min_disp * 9
# num_disp = max_disp - min_disp # Needs to be divisible by 16
# #Create Block matching object. 
# stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
#  numDisparities = num_disp,
#  blockSize = 5,
#  uniquenessRatio = 5,
#  speckleWindowSize = 5,
#  speckleRange = 5,
#  disp12MaxDiff = 1,
#  P1 = 8*3*win_size**2,#8*3*win_size**2,
#  P2 =32*3*win_size**2) #32*3*win_size**2)
# #Compute disparity map
# print ("\nComputing the disparity  map...")
# disparity_map = stereo.compute(img1, img2)

# #Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
# plt.imshow(disparity_map,'gray')
# plt.show()
