import numpy as np
import cv2

VERSION = 13
SET = 1
#Specify image paths
img1_path = f'3D_v{VERSION}/set{SET}/cam1.jpg'
img2_path = f'3D_v{VERSION}/set{SET}/cam2.jpg'
#Load pictures
right = cv2.imread(img1_path)
left = cv2.imread(img2_path)

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


right = downsample_image(right,3)
left = downsample_image(left,3)



fx = 1391       # lense focal length
baseline = 173   # distance in mm between the two cameras
disparities = 64  # num of disparities to consider
block = 15        # block size to match
units = 0.001     # depth units

disparity = np.zeros(shape=left.shape).astype(float)
for i in range(block, left.shape[0] - block - 1):
    for j in range(block + disparities, left.shape[1] - block - 1):
        ssd = np.empty([disparities, 1])

        # calc SSD at all possible disparities
        l = left[(i - block):(i + block), (j - block):(j + block)]
        for d in range(0, disparities):
            r = right[(i - block):(i + block), (j - d - block):(j - d + block)]
            ssd[d] = np.sum((l[:,:]-r[:,:])**2)

        # select the best match
        disparity[i, j] = np.argmin(ssd)

# Convert disparity to depth
depth = np.zeros(shape=left.shape).astype(float)
depth[disparity > 0] = (fx * baseline) / (units * disparity[disparity > 0])

cv2.imshow('disp', disparity)
cv2.waitKey(0)