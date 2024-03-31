import numpy as np
import tensorflow
import math
import time
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
cap =cv2.VideoCapture(0)
detector =HandDetector(maxHands=2)
offset = 20
imgSize=300
counter =0
folder = "Numbers/0"
while True:
    success, img =cap.read()
    hands, img= detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h=hand['bbox']
        imgWhite =np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop=img[y - offset:y+h+offset, x-offset:x+w+offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        if aspectRatio >1:
            k = imgSize/h
            wCal= math.ceil(k*w)
            imgResize =cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wgap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wgap:wCal+wgap] = imgResize

        else:
            k = imgSize/w
            hCal= math.ceil(k*h)
            imgResize =cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hgap = math.ceil((imgSize-hCal)/2)
            imgWhite[hgap:hCal+hgap, :] = imgResize


        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print((counter))


