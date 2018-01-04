import cv2
import numpy as np
import matplotlib.pyplot as plt

from loader import *
from color import *
from gradient import *
from sklearn.preprocessing import StandardScaler

def single_img_features(img, color_space='RGB', spatial_size=(32, 32), 
                         hist_bins=32, hist_range=(0, 256), 
                         orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                         spatial_feat=True, hist_feat=True, hog_feat=True):
    feature_image = to_colorspace(img, color_space)

    img_features = []

    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        img_features.append(hist_features)
    
    if hog_feat:
        if hog_channel == 'ALL' or hog_channel == 'A':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_channel = int(hog_channel)
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                pix_per_cell, cell_per_block, vis=False, feature_vec=True)            
        img_features.append(hog_features)
    
    return np.concatenate(img_features)

def extract_features(items, *args, **kw):
    features_list = []
    
    for item in items:
        img = load(item)
        features = single_img_features(img, *args, **kw)
        features_list.append(features)

    X = np.vstack(features_list).astype(np.float64)
    return X



def extract_hog_features(items, color_space='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
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
    features_list = extract_features(items, color_space='HSV', spatial_feat=False, hist_feat=False, hog_feat=True)

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
    plot_features(samples(500))



if __name__ == '__main__':
    main(sys.argv[1:])