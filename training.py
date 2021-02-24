import numpy as np
import cv2
import glob
import timeit
import os
import pickle
from joblib import dump, load
import pandas as pd
from sklearn import svm
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction import image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from phase_extraction import  residual_and_cut, feature_extraction

current_path = os.getcwd()
path_image_live = current_path + "\\dataset\\face_live_crop\\"
path_image_phone = current_path+ "\\dataset\\face_fake_crop\\"

# load training set live face with label = 0
data_live = []
class_live = []
path_live = glob.glob(path_image_live+"*.jpg")

for i in range(len(path_live)):
    img = cv2.imread(path_live[i])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img,(150,150))
    height_im, width_im = img.shape
    imtemp = img*2
    
    central_pixels_vec1,central_pixels_vec2,patches_imdiff1,patches_imdiff2 = \
        residual_and_cut(img,imtemp,height_im,width_im)

    corrs1,corrs2 = feature_extraction(central_pixels_vec1,central_pixels_vec2,\
        patches_imdiff1,patches_imdiff2)

    corrs_all = np.concatenate((corrs1,corrs2),axis=-1)
    data_live.append(corrs_all)
    class_live.append(0)

data_live = np.array(data_live)
class_live = np.array(class_live)

#load training set fake face with label = 1
data_phone = []
class_phone = []
path_phone = glob.glob(path_image_phone+"*.jpg")

for i in path_phone:
    img = cv2.imread(i)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img,(150,150))
    height_im, width_im = img.shape
    imtemp = img*2
    
    central_pixels_vec1,central_pixels_vec2,patches_imdiff1,patches_imdiff2 = \
        residual_and_cut(img,imtemp,height_im,width_im)

    corrs1,corrs2 = feature_extraction(central_pixels_vec1,central_pixels_vec2,\
        patches_imdiff1,patches_imdiff2)

    corrs_all = np.concatenate((corrs1,corrs2),axis=-1)
    data_phone.append(corrs_all)
    class_phone.append(1)

data_phone = np.array(data_phone)
class_phone = np.array(class_phone)

# Concat hai tap du liệu thành một tập duy nhất
dataset = np.concatenate((data_live,data_phone),axis=0)
class_ = np.concatenate((class_live,class_phone))
dataset = np.squeeze(dataset)

X_train, X_test, y_train, y_test = train_test_split(dataset, class_, test_size=0.2, shuffle=True)

#pd.DataFrame(X_train).to_csv("file_to_fittransfrom.csv",index=False)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
#model = pickle.load(open('model_train_spoof_bigdataset.pkl', 'rb'))
model = svm.SVC(kernel="rbf",gamma=1,class_weight={0:4,1:2})
model.fit(X_train, y_train)

predicted_test = model.predict(X_test)
print("Accuracy on testing:", metrics.accuracy_score(y_test, predicted_test)*100)

cm = confusion_matrix(y_test, predicted_test)
print("Confuse matrix on testing\n", cm)

predicted_train = model.predict(X_train)
print("Accuracy on training:", metrics.accuracy_score(y_train, predicted_train)*100)

cm = confusion_matrix(y_train, predicted_train)
print("Confuse matrix on training\n", cm)

# Save model
pickle.dump(model, open('SVM_model_cameraip.pkl', 'wb'))
# Save sc
pickle.dump(sc, open("scaler.pkl","wb"))