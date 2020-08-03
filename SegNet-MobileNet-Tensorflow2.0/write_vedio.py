import numpy as np
import cv2
from networks.segnet import SegNet_Mobile
from PIL import Image
import numpy as np
import random
import copy
import os
import time

cap = cv2.VideoCapture('/home/fmc/WX/Segmentation/SegNet-Mobile-tf2/1.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (960, 544))

while(cap.isOpened()):
    ret, img = cap.read()
    out.write(img)
    cv2.imshow('image', img)
    k = cv2.waitKey(30)
    # q键退出
    if (k & 0xff == ord('q')):
        break

cap.release()
out.relaase()
cv2.destroyAllWindows()