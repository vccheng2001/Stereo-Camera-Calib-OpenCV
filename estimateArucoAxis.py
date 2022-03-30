import cv2 
import numpy as np
import yaml
import cv2.aruco as aruco


TYPE = "CHESSBOARD"
VERSION = 2
print(f'****** ESTIMATING AXIS ON {TYPE} V{VERSION} *********\n')

##  SET UP ARUCO 
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)

# Creating a theoretical board we'll use to calculate marker positions
board = aruco.GridBoard_create(
    markersX=5,
    markersY=7,
    markerLength=0.04,
    markerSeparation=0.01,
    dictionary=aruco_dict)

arucoParams = aruco.DetectorParameters_create()

if TYPE == "CHESSBOARD":
    d = {} 
    with open(f'CHESSBOARD/calib_v{VERSION}.yaml') as file:
        documents = yaml.full_load(file)

        for item, doc in documents.items():
            d[item] = np.array(doc) 

    camera_matrix = d['camera_matrix']
    dist_coeffs = d['dist_coeff']
    
else:
    camera_matrix, dist_coeffs = np.load('CHESSBOARD/calibration_v2.yaml', allow_pickle=True)




videoFile = 'arucoOut.avi'
# cap = cv2.VideoCapture(videoFile)
cap = cv2.VideoCapture(1)

while(True):
    ret, frame = cap.read() # Capture frame-by-frame
    cv2.imshow('frame', frame)
    cv2.waitKey(1000)
    if ret == True:
        frame_remapped = frame 
        # frame_remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)     # for fisheye remapping
        frame_remapped_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_remapped_gray, aruco_dict, parameters=arucoParams)  # First, detect markers
        aruco.refineDetectedMarkers(frame_remapped_gray, board, corners, ids, rejectedImgPoints)

        if np.all(ids != None): # if there is at least one marker detected
            im_with_aruco_board = aruco.drawDetectedMarkers(frame_remapped, corners, ids, (0,255,0))
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, None, None)  # posture estimation from a diamond
            # print('trans', tvec)
            if retval != 0:
                im_with_aruco_board = aruco.drawAxis(im_with_aruco_board, camera_matrix, dist_coeffs, rvec, tvec, 100)  # axis length 100 can be changed according to your requirement
            else:
                print('retval 0')
        else:
            im_with_aruco_board = frame_remapped

        cv2.imshow("arucoboard", im_with_aruco_board)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()   # When everything done, release the capture
cv2.destroyAllWindows()