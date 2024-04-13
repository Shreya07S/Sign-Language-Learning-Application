import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20 
imgSize = 300


folder = "A"

counter = 0

while True:
    success, img= cap.read()
    hands , img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w , h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255



        imgCrop = img[y- offset:y+h+offset, x - offset:x+w + offset]
        
        imgCropShape = imgCrop.shape
        
        
        aspectRatio = h/w
        if aspectRatio >1:
            k = imgSize/h
            wCal = (math.ceil(k*w))
            imgResize = cv2.resize(imgCrop,(wCal , imgSize))
            imgResizeshape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[: , wGap:wCal+wGap] = imgResize

        else:
            k = imgSize/w
            hCal = (math.ceil(k*h))
            imgResize = cv2.resize(imgCrop,(imgSize, hCal))
            imgResizeshape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap ,:] = imgResize

        
        
        cv2.imshow("imgCrop" , imgCrop)
        cv2.imshow("imgWhite" , imgWhite)

    cv2.imshow("Image" , img)
    key = cv2.waitKey(1)

    if key == ord("S"):
        
       # cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        try:
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print("Image saved successfully.")
        except Exception as e:
            print("Error saving image:", e)
        counter += 1
        print(counter)
 