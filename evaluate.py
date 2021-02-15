import numpy as np
from mtcnn import MTCNN
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import mtcnn
import argparse
import cv2
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import pandas as pd
import timeit

def residual_and_cut(im, imtemp):
    # residue images
    imdiff1 = imtemp - cv2.filter2D(im.astype(np.float32),-1,h1)
    imdiff2 = imtemp - cv2.filter2D(im.astype(np.float32),-1,h2)

    # trimming residues images
    imdiff1 = imdiff1[1:height_im-1,1:width_im-1]
    imdiff2 = imdiff2[1:height_im-1,1:width_im-1]

    # extract the vector for the central pixel
    height_imdiff,width_imdiff = imdiff1.shape
    
    # for residue image 1
    central_pixels1 = imdiff1[half_window:height_imdiff-1-\
        half_window,half_window:width_imdiff-1-half_window]
    central_pixels1 = central_pixels1.T
    central_pixels_vec1 = central_pixels1.flatten(order="F")

    # for residue image 2
    central_pixels2 = imdiff2[half_window:height_imdiff-1-\
        half_window,half_window:width_imdiff-1-half_window]
    central_pixels2 = central_pixels2.T
    central_pixels_vec2 = central_pixels2.flatten(order="F")
    
    return central_pixels_vec1,central_pixels_vec2,imdiff1,imdiff2

def feature_extraction(central_pixels_vec1,central_pixels_vec2,\
    imdiff1,imdiff2,height_imdiff,width_imdiff):
    counter = 0
    for kk in range(corr_size):
        for jj in range(kk):
            if kk==3 and jj==3:
                continue
                
            current_pixels1 = imdiff1[jj:height_imdiff-corr_size+jj,kk:width_imdiff-corr_size+kk]
            current_pixels1 = current_pixels1.T
            current_pixels_vec1 = current_pixels1.flatten(order="F")
            mean_central_pixels_vec1 = central_pixels_vec1.mean()
            
            mean_current_pixels_vec1 = current_pixels_vec1.mean()
            tuso = np.sum((central_pixels_vec1-mean_central_pixels_vec1)*\
                (current_pixels_vec1-mean_current_pixels_vec1))
            mauso = np.sqrt(np.sum(((central_pixels_vec1-mean_central_pixels_vec1)**2))) * \
                np.sqrt(np.sum(((current_pixels_vec1-mean_current_pixels_vec1)**2)))
            corrs1[0,counter] = tuso/mauso
        
            current_pixels2 = imdiff2[jj:height_imdiff-corr_size+jj,kk:width_imdiff-corr_size+kk]
            current_pixels2 = current_pixels2.T
            current_pixels_vec2 = current_pixels2.flatten(order="F")
            mean_central_pixels_vec2 = central_pixels_vec2.mean()
            
            mean_current_pixels_vec2 = current_pixels_vec2.mean()
            tuso1 = np.sum((central_pixels_vec2-mean_central_pixels_vec2)*\
                (current_pixels_vec2-mean_current_pixels_vec2))
            mauso1 = np.sqrt(np.sum(((central_pixels_vec2-mean_central_pixels_vec2)**2))) * \
                np.sqrt(np.sum(((current_pixels_vec2-mean_current_pixels_vec2)**2)))
            corrs2[0,counter] = tuso1/mauso1
            counter = counter + 1
    
    return corrs1,corrs2

h1 = np.array([[0,0,0],[1,0,1],[0,0,0]])
h2 = np.array([[0,1,0],[0,0,0],[0,1,0]])

corr_size = 5
half_window = int((corr_size-1) / 2)
corrs1 = np.zeros((1,14))
corrs2 = np.zeros((1,14))

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image")
args = vars(ap.parse_args())
image = cv2.imread(args["input"])

detector = MTCNN()
result = detector.detect_faces(image)
box = result[0]["box"]
img = image[box[1] : box[1]+box[3],box[0]:box[0]+box[2]]
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

height_im, width_im = img.shape
imtemp = img*2

model = pickle.load(open('model.pkl', 'rb'))
sc = StandardScaler()

start1 = timeit.default_timer()
central_pixels_vec1,central_pixels_vec2,imdiff1,imdiff2 = residual_and_cut(img,imtemp)
height_imdiff,width_imdiff = imdiff1.shape
corrs1,corrs2 = feature_extraction(central_pixels_vec1,central_pixels_vec2,imdiff1,\
    imdiff2,height_imdiff,width_imdiff)
corrs_all = np.concatenate((corrs1,corrs2),axis=-1)
stop1 = timeit.default_timer()

df = pd.read_csv("dataset_spoof_face.csv",index_col=0)
df.pop("28")
data = np.array(df)
sc.fit_transform(data)
corrs_all = sc.transform(corrs_all)
start2 = timeit.default_timer()
predicted = model.predict(corrs_all)
stop2 = timeit.default_timer()

if predicted==0:
    print("Đây là hình ảnh live")
elif predicted==1:
    print("Đây là hình ảnh giả mạo")

print("inference time of the extractor:{0:10.5f}s".format(stop1-start1))
print("inference time of the classifier:{0:10.5f}s".format(stop2-start2))