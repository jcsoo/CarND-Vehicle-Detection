import cv2
import numpy as np
import matplotlib.pyplot as plt

from loader import *
from color import *
from gradient import *
from features import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

def item_labels(items):
    labels = [1 if item['tag'] == 'vehicle' else 0 for item in items]
    return np.array(labels)

def main(args):
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

    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])

if __name__ == '__main__':
    main(sys.argv[1:])