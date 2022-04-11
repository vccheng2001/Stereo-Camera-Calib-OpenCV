import numpy as np
import cv2 
import glob
import yaml
import os
''' Camera calibration 
Procedure: Take 10+ photos of chessboard from various viewpoints
to calibrate camera. 
Returns intrinsic matrix and calculates reprojection errors
'''

print('****** CALIBRATING CHESSBOARD FROM IMAGES *********\n')

TYPE="CHESSBOARD_TWOCAMS"
VERSION = 11

# ``columns`` and ``rows`` should be the number of inside corners in the
# chessboard's columns and rows. ``show`` determines whether the frames
# are shown while the cameras search for a chessboard.

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
CHECKERBOARD = (6,9) # cols, rows 
print('init check')


class Calibrator:
    def __init__(self, imageSize, cb_shape, cb_size):
        self.cb_shape = tuple([int(x) for x in cb_shape.split('x')])
        self.pattern_points = np.zeros((np.prod(self.cb_shape), 3), np.float32)
        self.pattern_points[:, :2] = np.indices(self.cb_shape).T.reshape(-1, 2)
        self.pattern_points *= cb_size

        self.imageSize = tuple([int(x) for x in imageSize.split('x')])
        self.alpha = -1

        self.term = (cv2.TERM_CRITERIA_EPS +
                     cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.arrays = None
        self.calibration = None

    def read_images(self, dir):
        print('dir', dir)

        assert os.path.isdir(dir+f'/CAM1_imgs_v{VERSION}') and os.path.isdir(dir+f'/CAM2_imgs_v{VERSION}')

        def find_corners(p):
            img = cv2.imread(p, 0)
            img = cv2.resize(img, self.imageSize)
            ret, corners = cv2.findChessboardCorners(
                img, self.cb_shape, cv2.CALIB_CB_FAST_CHECK)
            if ret and img.shape[::-1] == self.imageSize:
                cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), self.term)
                return [os.path.basename(p), self.pattern_points, corners]

        arr_left = np.array([find_corners(p)
                             for p in sorted(glob.glob(f"{dir}/CAM1_imgs_v{VERSION}/*.jpg"))], dtype='object')
        arr_left = arr_left[arr_left != None][0]

        print(arr_left)

        arr_right = np.array([find_corners(p)
                              for p in sorted(glob.glob(f"{dir}/CAM2_imgs_v{VERSION}/*.jpg"))], dtype='object')
        arr_right = arr_right[arr_right != None][0]

        # print(arr_left[:, 0])

        all_names = sorted(list(set(arr_left[:, 0]) & set(arr_right[:, 0])))

        def get_intersection(arr, all_names):
            return arr[np.isin(arr[:, 0], all_names)]

        arr_left = get_intersection(arr_left, all_names)
        arr_right = get_intersection(arr_right, all_names)

        self.arrays = [arr_left, arr_right]
        print(f'Found {len(arr_left)} images with chessboard')

    def calibrate_cameras(self):
        assert self.arrays
        matrix_left, distortion_left = cv2.calibrateCamera(
            self.arrays[0][:, 1], self.arrays[0][:, 2], self.imageSize, None, None)[1:3]
        matrix_right, distortion_right = cv2.calibrateCamera(
            self.arrays[0][:, 1], self.arrays[1][:, 2], self.imageSize, None, None)[1:3]

        # Do not need Apriltag
        rot_matrix, trans_vector = cv2.stereoCalibrate(
            self.arrays[0][:, 1], self.arrays[0][:, 2], self.arrays[1][:, 2],
            matrix_left, distortion_left,
            matrix_right, distortion_right,
            self.imageSize, flags=cv2.CALIB_FIX_INTRINSIC, criteria=self.term)[5:7]

        rect_left, rect_right, proj_left, proj_right, disparity, ROI_left, ROI_right = cv2.stereoRectify(
            matrix_left, distortion_left,
            matrix_right, distortion_right,
            self.imageSize, rot_matrix, trans_vector,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=self.alpha)

        self.calibration = {
            'general': {
                'rotation': rot_matrix,
                'translation': trans_vector,
                'disparity': disparity,
            },
            'left': {
                'matrix': matrix_left,
                'distortion': distortion_left,
                'rectification': rect_left,
                'projection': proj_left,
                'ROI': ROI_left,
            },
            'right': {
                'matrix': matrix_right,
                'distortion': distortion_right,
                'rectification': rect_right,
                'projection': proj_right,
                'ROI': ROI_right,
            }
        }

    def save(self, path):
        assert self.calibration


        with open(dest, "w") as f:
            yaml.dump(self.calibration, f)


size = '1920x1080'
dest = ''
cb_shape = '6x9' # cols, rows 
cb_size = 0.04
dir = f'{TYPE}_v{VERSION}'
dest = './{TYPE}/calib_v{VERSION}.yaml'
def main(dir, dest, size, cb_shape, cb_size):
    calibr = Calibrator(size, cb_shape, cb_size)
    calibr.read_images(dir)
    calibr.calibrate_cameras()
    calibr.save(dest)


if __name__ == "__main__":
    main(dir, dest, size, cb_shape, cb_size)