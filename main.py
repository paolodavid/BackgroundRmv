import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
imgBg = cv2.imread("Images/2.jpg")


listImg = os.listdir("Images")
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'Images/{imgPath}')
    imgList.append(img)

indexImg = 0

while True:
    _, frame = cap.read()

    imgOut = segmentor.removeBG(frame, imgList[indexImg], threshold=0.95)

    imgStacked = cvzone.stackImages([frame, imgOut], 2, 1)
    _, imgStacked = fpsReader.update(imgStacked, color=(0, 0, 255))

    cv2.imshow("Image", imgStacked)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImg>0:
            indexImg -= 1
    elif key == ord('d'):
        if indexImg < len(imgList)-1:
            indexImg += 1
    elif key == ord('q'):
        break