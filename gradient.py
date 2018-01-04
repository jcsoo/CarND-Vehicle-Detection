import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from loader import *

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features

def main(args):
    for item in samples(5):
        path, tag = item['path'], item['tag']
        img = load_image(path)[:,:,0]


        features, hog_image = get_hog_features(img, 9, 8, 2, True, True)

        
        fig = plt.figure(figsize=(12,3))
        plt.subplot(131)
        plt.imshow(img)
        plt.title('%s' % tag)    

        plt.subplot(132)
        plt.imshow(hog_image)
        plt.title('features')

        plt.subplot(133)
        plt.plot(features.ravel())
        

        plt.show()
        



if __name__ == '__main__':
    main(sys.argv[1:])