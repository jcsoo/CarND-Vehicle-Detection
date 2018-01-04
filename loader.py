import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os, glob, random

def load_png(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def load_jpg(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def load_image(path):
    ext = os.path.splitext(path)[1]
    if ext == '.png':
        return load_png(path)
    elif ext == '.jpg' or ext == 'jpeg':
        return load_jpg(path)
    else:
        raise Exception("Cannot load image with extension %s", ext)
        
def load(item):
    if item.get('img') is None:
        item['img'] = load_image(item['path'])
    return item['img']

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
    items = samples()

    fig, axs = plt.subplots(8, 8, figsize=(12, 12))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()

    for i in np.arange(64):
        item = random.choice(items)
        img = load_image(item['path'])
        axs[i].axis('off')
        axs[i].set_title(item['tag'], fontsize=10)
        axs[i].imshow(img)

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])