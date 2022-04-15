  
import cv2
import os
   

save = False
TYPE = "ARUCO_2CAMS"
VERSION = 2
CAM_NUM = 1
print(f'****** CAPTURING LIVE VIDEO FOR {TYPE} *********\n')
print(f'****** make sure to specify TYPE!!!  *********\n')

# Create an object to read 
# from camera
if CAM_NUM == 2:
    video = cv2.VideoCapture(0) # 0: long wire (cam2 left)
else:
    video = cv2.VideoCapture(2) # camera 1 right
   

# We need to check if camera
# is opened previously or not
if (video.isOpened() == False): 
    print("Error reading video file")
  
# We need to set resolutions.e
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))
print(f'width:{frame_width},height:{frame_height}')
   
size = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.

result = cv2.VideoWriter(f'{TYPE}/CAM{CAM_NUM}_vid_v{VERSION}.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
    
i = 0


if not os.path.exists(f'{TYPE}/'):
    os.mkdir(f'{TYPE}/')

if not os.path.exists(f'{TYPE}/CAM{CAM_NUM}_imgs_v{VERSION}/'):  
    os.mkdir(f'{TYPE}/CAM{CAM_NUM}_imgs_v{VERSION}/')

while(True):
    ret, frame = video.read()
  
    if ret == True: 
        
        # Write the frame into the
        # file 'filename.avi'

        if save: 
            result.write(frame)
            name = f'{i}.jpg'
            if i % 10 == 0: # dont save all 
                cv2.imwrite(os.path.join(f'{TYPE}/CAM{CAM_NUM}_imgs_v{VERSION}/', name),frame)
        i+=1
  
        # Display the frame
        # saved in the file
        cv2.imshow('Frame', frame)
  
        # Press S on keyboard 
        # to stop the process
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
  
    # Break the loop
    else:
        break
  
# When everything done, release 
# the video capture and video 
# write objects
video.release()
result.release()
    
# Closes all the frames
cv2.destroyAllWindows()

if save:
    print("The video was successfully saved")
else:
    print("Nothing saved")

print('****** DONE CAPTURING LIVE VIDEO  *********\n')
