import cv2
import numpy as np
import matplotlib.pyplot as plt

from loader import *
from color import *
from gradient import *
from features import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV

def item_labels(items):
    labels = [1 if item['tag'] == 'vehicle' else 0 for item in items]
    return np.array(labels)



def classify_color(args):
    count = int(args[0])
    spatial = int(args[1])
    histbin = int(args[2])
    
    print(count, spatial, histbin)
    items = samples(count)

    features = scale_features(extract_features(items, spatial_size=(spatial, spatial), hist_bins=(histbin, histbin)))
    labels = item_labels(items)

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=rand_state)

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    # Train the SVC
    svc.fit(X_train, y_train)
    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))

def classify_hog(args):
    count = int(args[0])

    colorspace = args[1] # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = args[2] # Can be 0, 1, 2, or "ALL"    
    
    print(count)
    items = samples(count)

    features = scale_features(extract_hog_features(items, colorspace, orient, pix_per_cell, cell_per_block, hog_channel))
    labels = item_labels(items)

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=rand_state)

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    # Train the SVC
    svc.fit(X_train, y_train)
    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))

def test_hog():
    count = 500
    orient = 9

    items = samples(count)

    for colorspace in COLOR_SPACES:
        for pix_per_cell in [4, 8]:
            for cell_per_block in [2, 4]:
                for hog_channel in [0, 1, 2, 'ALL']:
                    features = scale_features(extract_hog_features(items, colorspace, orient, pix_per_cell, cell_per_block, hog_channel))
                    labels = item_labels(items)

                    rand_state = np.random.randint(0, 100)
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=0.2, random_state=rand_state)

                    svc = LinearSVC()
                    svc.fit(X_train, y_train)
                    score = svc.score(X_test, y_test)
                    
                    print(colorspace, orient, pix_per_cell, cell_per_block, hog_channel, score)


def main(args):
    test_hog()

if __name__ == '__main__':
    main(sys.argv[1:])