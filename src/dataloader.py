import os
import glob
import cv2
import pickle
import numpy as np
from phase_extraction import FaceExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataLoader(object):

    def __init__(self, path_image_live, path_image_phone) -> None:
        self.face_extract = FaceExtractor()
        self.path_live = glob.glob(os.getcwd() + path_image_live + "*.jpg")
        self.path_phone = glob.glob(os.getcwd() + path_image_phone + "*.jpg")
        self.sc = StandardScaler()

    def load_extract_dataset(self, path_image, type):
        data_image = []
        class_image = []

        for i in range(len(path_image)):
            img = cv2.imread(path_image[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height_im, width_im = img.shape
            imtemp = img * 2
            
            central_pixels_vec1,central_pixels_vec2,patches_imdiff1,patches_imdiff2 = \
                self.face_extract.residual_and_cut(img, imtemp, height_im, width_im)
            
            corrs1, corrs2 = self.face_extract.feature_extraction(central_pixels_vec1,central_pixels_vec2,\
                patches_imdiff1,patches_imdiff2)
            
            corrs_all = np.concatenate((corrs1,corrs2), axis=-1)
            data_image.append(corrs_all)

            if type == "real":
                class_image.append(0)
            elif type == "fake":
                class_image.append(1)
        
        data_image = np.array(data_image)
        class_image = np.array(class_image)
        
        return data_image, class_image
    
    def process(self):
        self.data_live, self.class_live = self.load_extract_dataset(self.path_live, "real")
        self.data_fake, self.class_fake = self.load_extract_dataset(self.path_phone, "fake")
        dataset_image = np.concatenate((self.data_live, self.data_fake), axis=0)
        dataset_class = np.concatenate((self.class_live, self.class_fake))
        dataset_image = np.squeeze(dataset_image)
        
        x_train, x_test, y_train, y_test = train_test_split(dataset_image, dataset_class, test_size=0.2, shuffle=True, random_state=1)
        self.sc.fit(x_train)
        x_train = self.sc.transform(x_train)
        x_test = self.sc.transform(x_test)
        pickle.dump(self.sc, open("../models/scaler.pkl", "wb"))
        return (x_train, x_test, y_train, y_test)