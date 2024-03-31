import numpy as np
import tensorflow
import math
import time
import cv2
import cvzone
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
cap =cv2.VideoCapture(0)
detector =HandDetector(maxHands=2)
clas= Classifier("model/keras_model.h5","model/labels.txt")
offset = 20
imgSize=300
folder = "data/C"
counter = 0
labels = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
while True:
    success, img =cap.read()
    imgout=img.copy()
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
            prediction, index = clas.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize/w
            hCal= math.ceil(k*h)
            imgResize =cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hgap = math.ceil((imgSize-hCal)/2)
            imgWhite[hgap:hCal+hgap, :] = imgResize
            prediction, index = clas.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        cv2.rectangle(imgout, (x - offset, y-offset-50),
                     (x - offset+90, y-offset-50+50), (255,0,255),cv2.FILLED)
        cv2.putText(imgout,labels[index],(x,y-25),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgout,(x-offset, y-offset),
                      (x+w+offset,y+h+offset),(255,0,255),4)

        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)
    cv2.imshow("Image",imgout)
    cv2.waitKey(1)



