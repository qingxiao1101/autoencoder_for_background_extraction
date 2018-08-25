import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from skimage import io, transform

class Data():
    """
    Prepare data for training and testing 
    """
    def __init__(self):
        pass
    
    def get_image(self, file_path = './train_image'):
        img_names = [x for x in os.listdir(file_path) if x.split('.')[-1] in 'jpg|png']
        img_names = sorted(img_names, key = lambda x: int(x.split('.')[-2][2:]))
        img_paths = [os.path.join(file_path, x) for x in img_names]

        for img_path in img_paths:
            img = io.imread(img_path, as_grey = True)
            img = transform.resize(img, (300, 480))
            if img_paths.index(img_path) == 0:
                img_matrix = img[np.newaxis, :]
            else:
                img_matrix = np.concatenate((img_matrix, img[np.newaxis,:]), axis = 0)

        return img_matrix

if __name__ == '__main__':
    data = Data()
    data.get_image()
