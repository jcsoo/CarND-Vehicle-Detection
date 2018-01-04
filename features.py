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
        path = item['path']
        tag = item['tag']
        img = load_image(path)
        features = extract_img_features(img)
        features_list.append(features)

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