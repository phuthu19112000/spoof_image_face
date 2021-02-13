import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import scipy
import random
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import timeit

path_image_live = "E:/Task_edsolab/crop_image_live/"
path_image_phone = "E:/Task_edsolab/crop_phone_spoof/"

# 2 bộ lọc được sử dụng tính toán residual image
h1 = np.array([[0,0,0],[1,0,1],[0,0,0]])
h2 = np.array([[0,1,0],[0,0,0],[0,1,0]])

# Kích thước patch image
corr_size = 5
half_window = int((corr_size-1) / 2)

# 2 vector đặc trưng kết quả của quá trình thực hiện quá trính tương quan
corrs1 = np.zeros((1,14))
corrs2 = np.zeros((1,14))

# Hàm tính toám quá trình trích xuất 2 vector đặc trưng
def feature_extraction(central_pixels_vec1,central_pixels_vec2,\
    imdiff1,imdiff2,height_imdiff,width_imdiff):
    counter = 0
    for kk in range(corr_size):
        for jj in range(kk):
            if kk==3 and jj==3:
                continue

            # Tính toán cor cho residual image 1    
            current_pixels1 = imdiff1[jj:height_imdiff-corr_size+jj,\
                kk:width_imdiff-corr_size+kk]
            current_pixels1 = current_pixels1.T
            current_pixels_vec1 = current_pixels1.flatten(order="F")

            mean_central_pixels_vec1 = central_pixels_vec1.mean()
            mean_current_pixels_vec1 = current_pixels_vec1.mean()

            tuso = np.sum((central_pixels_vec1-mean_central_pixels_vec1)*\
                (current_pixels_vec1-mean_current_pixels_vec1))
            mauso = np.sqrt(np.sum(((central_pixels_vec1-mean_central_pixels_vec1)**2))) * \
                np.sqrt(np.sum(((current_pixels_vec1-mean_current_pixels_vec1)**2)))

            corrs1[0,counter] = tuso/mauso

            # Tính toán corr cho residual image 2
            current_pixels2 = imdiff2[jj:height_imdiff-corr_size+jj,\
                kk:width_imdiff-corr_size+kk]
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

def residual_and_cut(im, imtemp):
    # caculate residue images
    imdiff1 = imtemp - cv2.filter2D(im.astype(np.float32),-1,h1)
    imdiff2 = imtemp - cv2.filter2D(im.astype(np.float32),-1,h2)

    # trimming residues images
    imdiff1 = imdiff1[1:height_im-1,1:width_im-1]
    imdiff2 = imdiff2[1:height_im-1,1:width_im-1]

    # extract the vector for the central pixel
    height_imdiff,width_imdiff = imdiff1.shape
    
    # for residue image 1
    central_pixels1 = imdiff1[half_window:height_imdiff-1-half_window,half_window:\
        width_imdiff-1-half_window]
    central_pixels1 = central_pixels1.T
    central_pixels_vec1 = central_pixels1.flatten(order="F")

    # for residue image 2
    central_pixels2 = imdiff2[half_window:height_imdiff-1-half_window,half_window:\
        width_imdiff-1-half_window]
    central_pixels2 = central_pixels2.T
    central_pixels_vec2 = central_pixels2.flatten(order="F")
    
    return central_pixels_vec1,central_pixels_vec2,imdiff1,imdiff2

# load training set live face with label = 0
data_live = []
class_live = []
path_live = glob.glob(path_image_live+"*.jpg")
index = random.sample(range(len(path_live)),31000)

for i in index:
    path = path_live[i]
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height_im, width_im = img.shape
    imtemp = img*2
    
    central_pixels_vec1,central_pixels_vec2,imdiff1,imdiff2 = residual_and_cut(img,imtemp)
    height_imdiff,width_imdiff = imdiff1.shape
    corrs1,corrs2 = feature_extraction(central_pixels_vec1,central_pixels_vec2,\
        imdiff1,imdiff2,height_imdiff,width_imdiff)
    corrs_all = np.concatenate((corrs1,corrs2),axis=-1)
    data_live.append(corrs_all)
    class_live.append(0)

data_live = np.array(data_live)
class_live = np.array(class_live)

# load training set spoof LCD face with label = 1
data_phone = []
class_phone = []
path_phone = glob.glob(path_image_phone+"*.jpg")

for i in path_phone:
    img = cv2.imread(i)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height_im, width_im = img.shape
    imtemp = img*2
    
    central_pixels_vec1,central_pixels_vec2,imdiff1,imdiff2 = residual_and_cut(img,imtemp)
    height_imdiff,width_imdiff = imdiff1.shape
    corrs1,corrs2 = feature_extraction(central_pixels_vec1,central_pixels_vec2,\
        imdiff1,imdiff2,height_imdiff,width_imdiff)
    corrs_all = np.concatenate((corrs1,corrs2),axis=-1)
    data_phone.append(corrs_all)
    class_phone.append(1)

data_phone = np.array(data_phone)
class_phone = np.array(class_phone)

# Concat hai tap du liệu thành một tập duy nhất
dataset = np.concatenate((data_live,data_phone),axis=0)
class_ = np.concatenate((class_live,class_phone))

# Chuẩn hóa dữ liệu đầu vào về một khoảng nhất định
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Xây dựng model phân loại SVM với kernel rbf
model = svm.SVC(kernel="rbf",gamma=0.99)
X_train, X_test, y_train, y_test = train_test_split(dataset, class_, test_size=0.2, shuffle=True)
model.fit(X_train, y_train)

# predict
predicted = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, predicted))

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")

disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

# Testing time performance
img = cv2.imread(path_live[12323])
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
height_im, width_im = img.shape
imtemp = img*2

# Runtime for extract vector corr of residual face
start = timeit.default_timer()
central_pixels_vec1,central_pixels_vec2,imdiff1,imdiff2 = residual_and_cut(img,imtemp)
height_imdiff,width_imdiff = imdiff1.shape
corrs1,corrs2 = feature_extraction(central_pixels_vec1,central_pixels_vec2,\
    imdiff1,imdiff2,height_imdiff,width_imdiff)
corrs_all = np.concatenate((corrs1,corrs2),axis=-1)
stop = timeit.default_timer()
print('Time: ', stop - start)

# Runtime for classification spoof or live face
start = timeit.default_timer()
corrs_all = sc.transform(corrs_all)
predicted = clf.predict(corrs_all)
stop = timeit.default_timer()
print('Time: ', stop - start)

# Save model
from joblib import dump, load
dump(clf, 'SVM_spoof.joblib')
