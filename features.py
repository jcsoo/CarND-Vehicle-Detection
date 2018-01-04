import cv2
import numpy as np
import matplotlib.pyplot as plt

from loader import *
from color import *
from gradient import *
from sklearn.preprocessing import StandardScaler

def extract_img_features(img, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    feature_image = to_colorspace(img, cspace)

    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now

    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

    return np.concatenate((spatial_features, hist_features))

def scale_features(features_list):
    X = np.vstack(features_list).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    return scaled_X

def extract_features(items, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):    
    features_list = []
    
    for item in items:
        img = load(item)
        features = extract_img_features(img)
        features_list.append(features)

    X = np.vstack(features_list).astype(np.float64)
    return features_list



def extract_hog_features(items, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    features_list = []
    
    for item in items:        
        img = load(item)
        feature_image = to_colorspace(img, cspace)
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_channel = int(hog_channel)
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                pix_per_cell, cell_per_block, vis=False, feature_vec=True)            
        features_list.append(hog_features)

    X = np.vstack(features_list).astype(np.float64)
    return features_list    


def plot_features(items):
    features_list = extract_features(items)

    X = np.vstack(features_list).astype(np.float64)

    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)


    for i in range(5):
        ndx = np.random.randint(0, len(items))
        item = items[ndx]
        path = item['path']
        tag = item['tag']
        img = load_image(path)

        fig = plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(img)
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[ndx])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[ndx])
        plt.title('Normalized Features')
        fig.tight_layout()

        plt.show()


def main(args):
    plot_features(samples(64))



if __name__ == '__main__':
    main(sys.argv[1:])