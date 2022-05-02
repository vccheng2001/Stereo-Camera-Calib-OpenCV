import numpy as np
import cv2
import os
import argparse 

'''
Reads streams from left and right cameras simultaneously
Saves numbered frames into capture_right/, capture_left/
Also, saves videos as capture_right/video.avi and capture_left/video.avi

'''
def main(args):
    
    video_capture_1 = cv2.VideoCapture(2) # right cam (CAM 1)
    video_capture_2 = cv2.VideoCapture(0) # left cam (CAM 2)
    
    # get width, height 
    frame1_width, frame1_height = int(video_capture_1.get(3)), int(video_capture_1.get(4))
    frame2_width, frame2_height = int(video_capture_2.get(3)), int(video_capture_2.get(4))
    print(f'width1:{frame1_width},height1:{frame1_height}')
    print(f'width2:{frame2_width},height2:{frame2_height}')

    size = (frame1_width, frame1_height)

    i = 0


    if not os.path.exists('capture_right/'):
        os.mkdir('capture_right/')

    if not os.path.exists('capture_left/'):
        os.mkdir('capture_left/')

    # Video Writers (make sure same size)
    result1 = cv2.VideoWriter('capture_right/video.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)


    result2 = cv2.VideoWriter('capture_left/video.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)


    while True:
        # Capture frame-by-frame
        ret1, frame1 = video_capture_1.read()
        ret2, frame2 = video_capture_2.read()

        # display window
        if (ret1):
            cv2.imshow('Cam 1', frame1)
            cv2.imshow('Cam 2', frame2)

        if args.save: 
            result1.write(frame1)
            result2.write(frame2)
            name = f'{i}.jpg'
            if i % 10 == 0: # dont save all 
                print(f'Saving frame {i}')
                cv2.imwrite(os.path.join('capture_right/', name),frame1)
                cv2.imwrite(os.path.join('capture_left/', name), frame2)
        i+=1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture
    video_capture_1.release()
    video_capture_2.release()
    result1.release()
    result2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Draw Epipolar Lines")
    parser.add_argument("--save", type=bool, default=True, help="image captured by right camera")
    args = parser.parse_args()
    main(args)
    