import sys, os, json, pprint, pickle, time

from loader import *
from features import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV

def train_with_spec(spec):
    count = spec.get('count')

    car_items = vehicles(count)
    notcar_items = non_vehicles(count)

    items = []
    items.extend(car_items)
    items.extend(notcar_items)


    features_list = []
    for item in items:
        features_list.append(image_features(to_colorspace(load(item), spec.get('color_space', 'RGB')), spec))

    X = np.vstack(features_list).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)    

    # Define the labels vector
    y = np.hstack((np.ones(len(car_items)), np.zeros(len(notcar_items))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print(X_train.shape, X_train.dtype)

    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    # Train the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))    
    return svc, X_scaler


def train_with_file(spec_file):
    with open(spec_file) as f:
        clf, scl = train_with_spec(json.load(f))
    with open(spec_file.replace('.json','.clf'), 'wb') as f:
        pickle.dump(clf, f)
    with open(spec_file.replace('.json','.scl'), 'wb') as f:
        pickle.dump(scl, f)


def main(args):
    train_with_file(args[0])

if __name__ == '__main__':
    main(sys.argv[1:])