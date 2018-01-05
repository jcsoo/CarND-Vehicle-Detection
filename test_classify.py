import cv2
import numpy as np
import matplotlib.pyplot as plt
import time, json, pickle, pprint

from loader import *
from color import *
from gradient import *
from features import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog


def load_spec(spec_file):
    with open(spec_file) as f:
        spec = json.load(f)
    with open(spec_file.replace('.json','.clf'), 'rb') as f:
        clf = pickle.load(f)
    with open(spec_file.replace('.json','.scl'), 'rb') as f:
        scl = pickle.load(f)
    return spec, clf, scl

def test_spec(spec_file):
    spec, clf, scl = load_spec(spec_file)

    count = 32

    car_items = vehicles(count)
    notcar_items = non_vehicles(count)

    fig, axs = plt.subplots(8, 8, figsize=(12, 12))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()


    for i in np.arange(32):
        item = random.choice(car_items)
        img = load(item)
        features_list = []       
        features_list.append(image_features(to_colorspace(load(item), spec.get('color_space', 'RGB')), spec))

        test_features = scl.transform(np.array(features_list).reshape(1, -1))                
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        if prediction == 1:
            axs[i].set_title('car', fontsize=10)
        else:
            axs[i].set_title('not-car', fontsize=10)

        axs[i].axis('off')        
        axs[i].imshow(img)

    for i in np.arange(32, 64):
        item = random.choice(notcar_items)
        img = load(item)
        features_list = []       
        features_list.append(image_features(to_colorspace(load(item), spec.get('color_space', 'RGB')), spec))

        test_features = scl.transform(np.array(features_list).reshape(1, -1))                
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        if prediction == 1:
            axs[i].set_title('car', fontsize=10)
        else:
            axs[i].set_title('not-car', fontsize=10)

        axs[i].axis('off')        
        axs[i].imshow(img)            

    plt.show()    

def main(args):
    test_spec(args[0])

if __name__ == '__main__':
    main(sys.argv[1:])