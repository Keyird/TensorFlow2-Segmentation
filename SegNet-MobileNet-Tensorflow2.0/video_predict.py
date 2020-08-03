import numpy as np
import cv2
from networks.segnet import SegNet_Mobile
from PIL import Image
import numpy as np
import random
import copy
import os
import time

class_colors = [[0,0,0], [0,255,0]]
NCLASSES = 2
HEIGHT = 416
WIDTH = 416

model = SegNet_Mobile(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
model.load_weights("models/last1.h5")

cap = cv2.VideoCapture('/home/fmc/WX/Segmentation/SegNet-Mobile-tf2/1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (960, 544))
fps = 0.0

while(cap.isOpened()):

    t1 = time.time()
    ret, img = cap.read()

    #############################
    # 格式转变，BGRtoRGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转变成Image
    img = Image.fromarray(np.uint8(img))

    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((WIDTH, HEIGHT))
    img = np.array(img)

    img = img / 255
    img = img.reshape(-1, HEIGHT, WIDTH, 3)
    pr = model.predict(img)[0]

    pr = pr.reshape((int(HEIGHT / 2), int(WIDTH / 2), NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT / 2), int(WIDTH / 2), 3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))

    img = Image.blend(old_img, seg_img, 0.3)

    img = np.array(img)
    fps = (fps + (1. / (time.time() - t1))) / 2
    print("fps= %.2f" % (fps))
    img = cv2.putText(img, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #############################
    out.write(img)
    cv2.imshow('image', img)
    k = cv2.waitKey(10)
    #q键退出
    if (k & 0xff == ord('q')):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

"""
 old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((WIDTH, HEIGHT))
    img = np.array(img)
    print(img)
    img = img / 255
    img = img.reshape(-1, HEIGHT, WIDTH, 3)
    pr = model.predict(img)[0]

    pr = pr.reshape((int(HEIGHT / 2), int(WIDTH / 2), NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT / 2), int(WIDTH / 2), 3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))

    image = Image.blend(old_img, seg_img, 0.3)
 
"""