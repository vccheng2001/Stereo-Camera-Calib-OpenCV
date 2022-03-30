  
import cv2
import os

# USE THIS FOR MOBILE 
   
TYPE = "CHESSBOARD"
VERSION = 2
CAM_NUM = "M"
print(f'****** READING VIDEO INTO IMAGES FOR {TYPE} ON CAMERA {CAM_NUM}*********\n')
print(f'****** make sure to specify TYPE!!!  *********\n')

# Create an object to read 
# from camera


cap = cv2.VideoCapture(f'{TYPE}/CAM{CAM_NUM}_vid_v{VERSION}.mp4')

i = 0

if not os.path.exists(f'{TYPE}/'):
    os.mkdir(f'{TYPE}/')

if not os.path.exists(f'{TYPE}/CAM{CAM_NUM}_imgs_v{VERSION}/'):  
    os.mkdir(f'{TYPE}/CAM{CAM_NUM}_imgs_v{VERSION}/')
    
while(True):

    ret, frame = cap.read()
  
    if ret == True: 
        
        # Write the frame into the
        # file 'filename.avi'

        name = f'{i}.jpg'
        if i % 10 == 0: # dont save all 
            cv2.imwrite(os.path.join(f'{TYPE}/CAM{CAM_NUM}_imgs_v{VERSION}/', name),frame)
        i+=1
  
        # Display the frame
        # saved in the file
        # cv2.imshow('Frame', frame)
  
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
cap.release()
    
# Closes all the frames
cv2.destroyAllWindows()
   
print("The video was successfully saved")

print('****** DONE READING VIDEOS INTO IMAGES FOR {TYPE} ON CAMERA {CAM_NUM} *********\n')
