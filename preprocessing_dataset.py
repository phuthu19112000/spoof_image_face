import json
import cv2
import numpy as np
import os
import glob
from shutil import copy
from PIL import Image  
import PIL
import os.path
from os import path

# directory and file path
path_image_train = "E:/Task_edsolab/CelebA_Spoof/"
path_image_json = "E:/Task_edsolab/CelebA_Spoof/metas/intra_test/train_label.json"
path_image_class = "E:/Task_edsolab/CelebA_Spoof/metas/intra_test/train_label.txt"
path_image_7 = "E:/Task_edsolab/image_phone_spoof/"
crop_spoof_7 = "E:/Task_edsolab/crop_phone_spoof/"
crop_image_live = "E:/Task_edsolab/crop_image_live/"
path_image_live = "E:/Task_edsolab/image_live/"

# open file dataset CelebA-Spoof and read, reject characters for preparing
with open(path_image_json) as f:
    data = json.load(f)

with open(path_image_class,"r") as f:
    image_class = f.readlines()

for i in range(len(image_class)):
    image_class[i] = image_class[i].replace("\n","")
    image_class[i] = image_class[i].split(" ")
    image_class[i][1] = int(image_class[i][1])

keys_data = data.keys()
keys_data = list(keys_data)

# Filter spoof images throught LCD class
for i in range(len(keys_data)):

    if image_class[i][1] == 0:
        continue

    elif image_class[i][1] == 1:
        attribute = data[keys_data[i]]
        if attribute[40] == 7:
            file_image = path_image_train + keys_data[i]
            file_BB = path_image_train + keys_data[i].replace(".jpg","_BB.txt")
            copy(file_image,path_image_7)
            copy(file_BB,path_image_7)

# Crop image live face with bounding box and save for training dataset
name_path = glob.glob(path_image_live +'*.jpg')
name_box = glob.glob(path_image_live +'*.txt')

for i in range(len(name_path)):
    img = cv2.imread(name_path[i])
    real_h,real_w,deapth = img.shape
    with open(name_box[i],"r") as f:
        value = f.read()
    bbox = value.replace("\n","").split(" ")
    # Scale box_fake to box_real of images
    x1 = int(int(bbox[0]) * (real_w / 244))
    y1 = int(int(bbox[1]) * (real_h / 244))
    w1 = int(int(bbox[2]) * (real_w / 244))
    h1 = int(int(bbox[3]) * (real_h / 224))
    img_crop = img[y1:y1+h1,x1:x1+w1]
    if img_crop.size == 0:
        continue
    else:
        cv2.imwrite(crop_image_live+"{}.jpg".format(i),img_crop)

# loai bo bad image no condition
image_list = glob.glob(crop_image_live+"*.jpg")
for i in range(len(image_list)):
    image = cv2.imread(image_list[i])
    height,width,deapth = image.shape
    if height <= 50 or width <= 50:
        os.remove(image_list[i])
    else:
        continue

# Crop image spoof face and save for training dataset
name_path = glob.glob(path_image_7 +'*.jpg')
name_box = glob.glob(path_image_7 +'*.txt')

for i in range(len(name_path)):
    img = cv2.imread(name_path[i])
    real_h,real_w,deapth = img.shape
    with open(name_box[i],"r") as f:
        value = f.read()
    bbox = value.replace("\n","").split(" ")
    x1 = int(int(bbox[0]) * (real_w / 244))
    y1 = int(int(bbox[1]) * (real_h / 244))
    w1 = int(int(bbox[2]) * (real_w / 244))
    h1 = int(int(bbox[3]) * (real_h / 224))
    img_crop = img[y1:y1+h1,x1:x1+w1]
    if img_crop.size == 0:
        continue
    else:
        cv2.imwrite(crop_spoof_7+"{}.jpg".format(i),img_crop)

# loai bo bad image no condition
image_list = glob.glob(crop_spoof_7+"*.jpg")
for i in range(len(image_list)):
    image = cv2.imread(image_list[i])
    height,width,deapth = image.shape
    if height <= 50 or width <= 50:
        os.remove(image_list[i])
    else:
        continue