import cv2
import os
from cvzone.HandTrackingModule  import HandDetector
import mediapipe as mp


widht,height=1280,720

folderpath="C:/Users/Dhruv Chaudhary/Desktop/ml_projects/hand_gesture_recognizer/prsntion"
imagepath=sorted(os.listdir(folderpath),key=len) # itb will create a sorted list of paths 
# print(imagepath) # a list of image path
#setup camera

cap=cv2.VideoCapture(0) # it will capture the frame and save it to cap varible"
cap.set(3,widht)
cap.set(4,height)

imagenumber=0
hs,ws= int(120*1.2) , int(213*1.2)

detector=HandDetector(detectionCon=0.8 , maxHands=2)



while True:
    success , img=cap.read()  # captured frame will be readed by .read() function
    
    img=cv2.flip(img,1) # 1 means horizontal flip and 0 means vertocal flip

    path_full=os.path.join(folderpath,imagepath[imagenumber])
    imagecurrent=cv2.imread(path_full)
    
    hands , img=detector.findHands(img)
    
    if hands:
        hand=hands[0]
        finger=detector.fingersUp(hand)
        print(finger)
    # result=hands.process(img)
    # print(result)
    
    # adding webcam image to slides window
    imgsmall=cv2.resize(img,(ws,hs))
    h,w,_=imagecurrent.shape # it will retrieve the dimension of current image [ h=number of rows ] , [w= no of colums ] [_no of color channel ]
    imagecurrent[0:hs,w-ws:w]=imgsmall


    cv2.imshow("Image",img)
    cv2.imshow("slides",imagecurrent)  # show the captured image


    
    key=cv2.waitKey(1) & 0xff  # here key will store the value neterd by the keyboard
 
    if key==ord("q"):

        cv2.destroyAllWindows()
        break
