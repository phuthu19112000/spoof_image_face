import numpy as np
import cv2
from mtcnn import MTCNN
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from sklearn import metrics
import glob
import argparse
from phase_extraction import *
import timeit

model = pickle.load(open('SVM_model_cameraip.pkl', 'rb'))
sc = pickle.load(open('scaler.pkl', 'rb'))

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image")
args = vars(ap.parse_args())
image = cv2.imread(args["input"])

detector = MTCNN()
result = detector.detect_faces(image)
box = result[0]["box"]
img = image[box[1] : box[1]+box[3],box[0]:box[0]+box[2]]
img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

height_im, width_im = img.shape
imtemp = img*2

start1 = timeit.default_timer()  
central_pixels_vec1,central_pixels_vec2,patches_imdiff1,patches_imdiff2 = \
    residual_and_cut(img,imtemp,height_im,width_im)

corrs1,corrs2 = feature_extraction(central_pixels_vec1,central_pixels_vec2,\
    patches_imdiff1,patches_imdiff2)
corrs_all = np.concatenate((corrs1,corrs2),axis=-1)
stop1 = timeit.default_timer()

corrs_all = sc.transform(corrs_all)
start2 = timeit.default_timer()
predicted = model.predict(corrs_all)
stop2 = timeit.default_timer()

if predicted[0]==0:
    print("Đây là hình ảnh live")
elif predicted[1]==1:
    print("Đây là hình ảnh spoof")

print("inference time of the extractor:{0:10.5f}s".format(stop1-start1))
print("inference time of the classifier:{0:10.5f}s".format(stop2-start2))

