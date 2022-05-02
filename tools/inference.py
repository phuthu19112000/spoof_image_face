
import numpy as np
import cv2
from mtcnn import MTCNN
import pickle
import argparse
import timeit
from src.phase_extraction import FaceExtractor

def initialize(path_model, path_scaler):
    global model_anti
    global sc
    global face_detector
    global face_extractor

    model_anti = pickle.load(open(path_model, 'rb'))
    sc = pickle.load(open(path_scaler, 'rb'))
    face_detector = MTCNN()
    face_extractor = FaceExtractor()


def preprocess(img_face, imtemp, height_im, width_im ):

    central_pixels_vec1,central_pixels_vec2,patches_imdiff1,patches_imdiff2 = \
        face_extractor.residual_and_cut(img_face,imtemp,height_im,width_im)
    corrs1,corrs2 = face_extractor.feature_extraction(central_pixels_vec1,central_pixels_vec2,\
        patches_imdiff1,patches_imdiff2)
    corrs_all = np.concatenate((corrs1,corrs2),axis=-1)
    return corrs_all

def infer(image_path):

    image = cv2.imread(image_path)
    result_face = face_detector.detect_faces(image)
    box = result_face[0]["box"]
    img_face = image[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
    img_face = cv2.cvtColor(img_face,cv2.COLOR_BGR2GRAY)
    height_im, width_im = img_face.shape
    imtemp = img_face*2
    
    vector_feature = preprocess(img_face, imtemp, height_im, width_im)
    vector_feature = sc.transform(vector_feature)
    start = timeit.default_timer()
    predicted = model_anti.predict(vector_feature)
    stop = timeit.default_timer()
    print("inference time of the classifier:{0:10.5f}s".format(stop-start))
    if predicted[0]==0:
        print("Đây là hình ảnh live")
    elif predicted[1]==1:
        print("Đây là hình ảnh spoof")

def parser_arumment():

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to model")
    ap.add_argument("-s", "--scaler", required=True, help="path to scaler dataset")
    ap.add_argument("-i", "--input", required=True, help="path to input image")
    args = vars(ap.parse_args())
    return args

if __name__ == "__main__":
    args = parser_arumment()
    initialize(args["model"], args["scaler"])
    infer(args["input"])

    