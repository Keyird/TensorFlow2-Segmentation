# coding:utf-8
import os

imgs_path = '/home/fmc/WX/Segmentation/SegNet-Mobile-tf2/dataset/jpg'  # 图片文件存放地址
for files in os.listdir(imgs_path):
    print(files)
    image_name = files + ';' + files[:-4] + '.png'

    with open("train.txt", "a") as f:
        f.write(str(image_name) + '\n')
f.close()