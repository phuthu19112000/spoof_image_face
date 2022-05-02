import numpy as np
import cv2
from sklearn.feature_extraction import image

class FaceExtractor(object):

    def __init__(self) -> None:
        # 2 bộ lọc được sử dụng tính toán residual image
        self.h1 = np.array([[0,0,0],[1,0,1],[0,0,0]])
        self.h2 = np.array([[0,1,0],[0,0,0],[0,1,0]])
        # Kích thước patch image
        self.corr_size = 5
        
    def init_vector_feature(self):
        # 2 vector đặc trưng kết quả của quá trình thực hiện tính toán tương quan
        self.corrs1 = np.zeros((1,24))
        self.corrs2 = np.zeros((1,24))

    def residual_and_cut(self, im, imtemp, height_im, width_im):
        '''
        parameters:
            im: Hình ảnh đầu vào X
            imtemp: Hình ảnh X*2 dùng để tính Residual image
        '''
        # caculate residue images
        imdiff1 = imtemp - cv2.filter2D(im.astype(np.float32),-1,self.h1)
        imdiff2 = imtemp - cv2.filter2D(im.astype(np.float32),-1,self.h2)

        # trimming residues images
        imdiff1 = imdiff1[1:height_im-1,1:width_im-1]
        imdiff2 = imdiff2[1:height_im-1,1:width_im-1]

        # extract the vector for the central pixel
        height_imdiff,width_imdiff = imdiff1.shape
        
        # for residue image 1 extract each patch image 5x5
        patches_imdiff1 = image.extract_patches_2d(imdiff1,(5,5))
        central_pixels_vec1 = patches_imdiff1[:,2,2]

        # for residue image 2 extract each patch iamge 5x5
        patches_imdiff2 = image.extract_patches_2d(imdiff2,(5,5))
        central_pixels_vec2 = patches_imdiff2[:,2,2]
        
        return central_pixels_vec1,central_pixels_vec2,patches_imdiff1,patches_imdiff2


    def feature_extraction(self, central_pixels_vec1,central_pixels_vec2, patches_imdiff1,patches_imdiff2):
        '''
        parameters:
            central_pixels_vec1: vector trung tâm của các patch 5x5 từ residual 1
            central_pixels_vec2: vector trung tâm của các patch 5x5 từ residual 2
            patches_imdiff1: một mảng numpy có shape = (số patch residual1, 5, 5)
            patches_imdiff2: một mảng numpy có shape = (số patch residual2, 5, 5)
        '''
        central_pixels_vec1 = patches_imdiff1[:,2,2]
        central_pixels_vec2 = patches_imdiff2[:,2,2]
        self.init_vector_feature()

        counter = 0
        for kk in range(self.corr_size):
            for jj in range(self.corr_size):
                if kk==2 and jj==2:
                    continue

                # Tính toán cor cho residual image 1
                current_pixels_vec1 = patches_imdiff1[:,kk,jj]

                mean_central_pixels_vec1 = central_pixels_vec1.mean()
                mean_current_pixels_vec1 = current_pixels_vec1.mean()

                tuso = np.sum((central_pixels_vec1-mean_central_pixels_vec1)*\
                    (current_pixels_vec1-mean_current_pixels_vec1))

                mauso = np.sqrt(np.sum(((central_pixels_vec1-mean_central_pixels_vec1)**2))) * \
                    np.sqrt(np.sum(((current_pixels_vec1-mean_current_pixels_vec1)**2)))

                self.corrs1[0,counter] = tuso/mauso

                # Tính toán corr cho residual image 2
                current_pixels_vec2 = patches_imdiff2[:,kk,jj]

                mean_central_pixels_vec2 = central_pixels_vec2.mean()
                mean_current_pixels_vec2 = current_pixels_vec2.mean()

                tuso1 = np.sum((central_pixels_vec2-mean_central_pixels_vec2)*\
                    (current_pixels_vec2-mean_current_pixels_vec2))

                mauso1 = np.sqrt(np.sum(((central_pixels_vec2-mean_central_pixels_vec2)**2))) * \
                    np.sqrt(np.sum(((current_pixels_vec2-mean_current_pixels_vec2)**2)))

                self.corrs2[0,counter] = tuso1/mauso1
                counter = counter + 1
        
        return self.corrs1,self.corrs2