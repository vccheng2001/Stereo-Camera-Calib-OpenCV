import numpy as np
import cv2
import os

save = True
video_capture_1 = cv2.VideoCapture(2) # right cam (CAM 1)
video_capture_2 = cv2.VideoCapture(0) # left cam (CAM 2)

VERSION = 13
SET = 1
TYPE = f"3D_v{13}/set{SET}"

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


while True:
    # Capture frame-by-frame
    ret1, frame1 = video_capture_1.read()
    ret2, frame2 = video_capture_2.read()
    if i == 5: break

   

    if save: 
        print('saving')
        result1.write(frame1)
        result2.write(frame2)
        if (i+1) % 5 == 0: # dont save all 
            cv2.imwrite(os.path.join(f'{TYPE}', 'cam1.jpg'),frame1)
            cv2.imwrite(os.path.join(f'{TYPE}', 'cam2.jpg'), frame2)
    i+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_1.release()
video_capture_2.release()
# result1.releas)e()
# result2.release(
cv2.destroyAllWindows()