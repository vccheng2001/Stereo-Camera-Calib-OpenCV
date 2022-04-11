import numpy as np
import cv2
import os

save = True
video_capture_1 = cv2.VideoCapture(2) # right cam (CAM 1)
video_capture_2 = cv2.VideoCapture(0) # left cam (CAM 2)

VERSION = 11
TYPE = f"ARUCO_TWOCAMS_v{VERSION}"

print(f'****** CAPTURING TWO VIDEOS FOR {TYPE} *********\n')
  
# We need to set resolutions.e
# so, convert them from float to integer.
frame1_width, frame1_height = int(video_capture_1.get(3)), int(video_capture_1.get(4))
frame2_width, frame2_height = int(video_capture_2.get(3)), int(video_capture_2.get(4))
print(f'width1:{frame1_width},height1:{frame1_height}')
print(f'width2:{frame2_width},height2:{frame2_height}')

size = (frame1_width, frame1_height)

# Video Writers 
result1 = cv2.VideoWriter(f'{TYPE}/CAM1_vid_v{VERSION}.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)


result2 = cv2.VideoWriter(f'{TYPE}/CAM2_vid_v{VERSION}.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
i = 0


if not os.path.exists(f'{TYPE}/'):
    os.mkdir(f'{TYPE}/')

if not os.path.exists(f'{TYPE}/CAM1_imgs_v{VERSION}/'):  
    os.mkdir(f'{TYPE}/CAM1_imgs_v{VERSION}/')

if not os.path.exists(f'{TYPE}/CAM2_imgs_v{VERSION}/'):  
    os.mkdir(f'{TYPE}/CAM2_imgs_v{VERSION}/')


while True:
    # Capture frame-by-frame
    ret1, frame1 = video_capture_1.read()
    ret2, frame2 = video_capture_2.read()

    if (ret1):
        # Display the resulting frame
        cv2.imshow('Cam 1', frame1)
    if (ret2):
        # Display the resulting frame
        cv2.imshow('Cam 2', frame2)

    if save: 
        print('saving')
        result1.write(frame1)
        result2.write(frame2)
        name = f'{i}.jpg'
        if i % 10 == 0: # dont save all 
            cv2.imwrite(os.path.join(f'{TYPE}/CAM1_imgs_v{VERSION}/', name),frame1)
            cv2.imwrite(os.path.join(f'{TYPE}/CAM2_imgs_v{VERSION}/', name), frame2)
    i+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_1.release()
video_capture_2.release()
# result1.releas)e()
# result2.release(
cv2.destroyAllWindows()