import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os, glob, random

def load_png(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def load_image(path):
    ext = os.path.splitext(path)[1]
    if ext == '.png':
        return load_png(path)
    else:
        raise Exception("Cannot load image with extension %s", ext)
        
def vehicles():
    items = []
    for path in glob.glob("vehicles/*/*.png"):
        items.append({
            'path': path,
            'tag': 'vehicle',
        })
    return items
    

def non_vehicles():    
    items = []
    for path in glob.glob("non-vehicles/*/*.png"):
        items.append({
            'path': path,
            'tag': 'non-vehicle',
        })
    return items
    
def samples(count=None):
    items = []
    items.extend(vehicles())
    items.extend(non_vehicles())
    random.shuffle(items)
    if count is not None:
        items = items[:count]    
    return items

def test_split(items, test_size=0.1):
    items_len = len(items)
    train_len = items_len - int(items_len * test_size)
    return items[:train_len], items[train_len:]

def main(args):
    (train, test) = test_split(samples(1000))
    for item in train[:5]:
        path = item['path']
        cv2.imshow(item['tag'], load_image(path))
        key = cv2.waitKey()
        if key == 27:
            return


if __name__ == '__main__':
    main(sys.argv[1:])