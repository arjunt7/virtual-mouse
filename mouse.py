import cv2
import numpy as np
import time 
from cvzone.HandTrackingModule import HandDetector
import pyautogui
from time import sleep
#import autopy

cam_W, cam_H = 640, 480
color_fps = (0,255,0)
frame_reduced = 120
smooth_value = 2

pTime = 0 # previous time
prev_x , prev_y = 0,0
curr_x , curr_y = 0,0

screenWidth, screenHeight = pyautogui.size() # Get the size of the screen


cap = cv2.VideoCapture(0) #input feed, index 0 -> default camera (indices 1,2 for other cameras)
cap.set(3,cam_W) #width -> 3
cap.set(4,cam_H)  #height -> 4
# creating a detector object 
detector = HandDetector(detectionCon = 0.6,maxHands=1 )
"""  HandDetector : creates a hand detection object 
     detectionCon : confidence threshhold (80% here)
"""
while True: 
    success, img  =cap.read()
    img = cv2.flip(img,1)  # 0-> vertical flip , 1-> horizontal flip
    hands,img  = detector.findHands(img)
    """ detector.findHands(img): Processes the frame (img) to detect hands
         hand: A list containing information about the detected hands (e.g., position of landmarks, bounding box, etc.).         
         img: The input image is returned with visual annotations, such as hand landmarks and bounding boxes, overlaid.      
    """
    #finding the tip of index and middle finger
    if hands: 
        hand1 = hands[0]
        lmList = hand1['lmList']
        index_finger= lmList[8][:2]
        middle_finger= lmList[12][:2]
        #print(index_finger, middle_finger)
    # now to find which fingers are up 
        fingerUp = detector.fingersUp(hand1)
        cv2.rectangle(img, (frame_reduced,frame_reduced),(cam_W-frame_reduced,cam_H- frame_reduced),(0,0,200),3 )
    # checking if only index finger up 
        if fingerUp[1] == 1 and fingerUp[2] == 0: # 1 -> index finger, 2-> middle finger
           # coordinates from cam to screen size
           mouse_X = np.interp(index_finger[0],(frame_reduced,cam_W-frame_reduced),(0,screenWidth))
           mouse_Y = np.interp(index_finger[1],(frame_reduced,cam_H-frame_reduced),(0,screenHeight))
           # smoothening the value of x and y before sending it formouse movement
           curr_x = prev_x + (mouse_X - prev_x)/smooth_value
           curr_y = prev_y + (mouse_Y - prev_y)/smooth_value
           # mouse movement
           pyautogui.moveTo(curr_x,curr_y)
           cv2.circle(img,(index_finger[0],index_finger[1]),10,(255,0,0),cv2.FILLED)
           prev_y = curr_y
           prev_x = curr_x
           # checking if both index and middle finger up setting ditance for click 
        if fingerUp[1] == 1 and fingerUp[2] == 1:
            l,line_info,_ = detector.findDistance(index_finger,middle_finger, img)
            if l<20: 
              cv2.circle(img,(line_info[4],line_info[5]),10,(0,255,0),cv2.FILLED) # 4,5are the index at which you find the center dot of the line joining the index and the middle finger 
              pyautogui.click()
              sleep(0.15)
    # for fps 
    cTime = time.time() # returns current time 
    fps = 1/(cTime - pTime)
    """cTime - pTime calculates the time elapsed between the current frame (cTime) and the previous frame (pTime).
       The reciprocal (1 / (cTime - pTime)) gives the frames per second (FPS) because FPS is the number of frames processed in one second. 
     """
    pTime = cTime # updating the previous time to current 
    cv2.putText(img,str(int(fps)),(20,50),cv2.Formatter_FMT_DEFAULT,3,color_fps,3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)