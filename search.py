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
from sklearn.model_selection import train_test_split, GridSearchCV

def load_spec(spec_file):
    with open(spec_file) as f:
        spec = json.load(f)
    with open(spec_file.replace('.json','.clf'), 'rb') as f:
        clf = pickle.load(f)
    with open(spec_file.replace('.json','.scl'), 'rb') as f:
        scl = pickle.load(f)
    return spec, clf, scl

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(shape, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def search_windows(img, windows, clf, scaler, spec):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))        
        #4) Extract features for that window using single_img_features()
        features = image_features(test_img, spec)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, y_start_stop, scale, clf, scl, spec):       
    ystart, ystop = y_start_stop
    spatial, hist, hog = spec.get('spatial'), spec.get('histogram'), spec.get('hog')

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = img_tosearch

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    shape = ctrans_tosearch.shape
    orient, pix_per_cell, cell_per_block = hog['orient'], hog['pix_per_cell'], hog['cell_per_block']


    # Define blocks and steps as above
    nxblocks = (shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image    

    

    hog_channels = []

    if hog:
        hog = hog.copy()
        for ch in hog.pop('channels'):
            hog_channels.append(get_hog_features(ctrans_tosearch[:,:,ch], vis=False, feature_vec=False, **hog))
    
    boxes = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            features = []


            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            if spatial:
                features.append(bin_spatial(subimg, **spatial))

            if hist:
                features.append(color_hist(subimg, **hist))

            if hog:
                hog_features = []
                for hc in hog_channels:
                    hog_features.append(hc[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())
                features.append(np.hstack(hog_features))

            # Scale features and make a prediction
            test_features = scl.transform(np.hstack(features).reshape(1, -1))    
            test_prediction = clf.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return boxes


def search(spec_file, paths):
    spec, clf, scl = load_spec(spec_file)
    pprint.pprint(spec)

    y_start_stop = [400, 700] # Min and max in y to search in slide_window()    
    sizes = [128, 96, 64]
    # scales = [1.0, 1.5, 2.0]
    scales = [1.5, 2.0]

    for path in paths:
        img = load_image(path)
        draw_img = img.copy()
        feature_img = to_colorspace(img, spec.get('color_space', 'RGB'))

        hot_windows = []

        if True:
            for scale in scales:
                hot_windows.extend(find_cars(feature_img, y_start_stop, scale, clf, scl, spec))
        else:
            for size in sizes:
                windows = slide_window(img.shape, x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=(size, size), xy_overlap=(0.5, 0.5))
                hot_windows.extend(search_windows(feature_img, windows, clf, scl, spec))

        draw_img = draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)                
        plt.imshow(draw_img)
        plt.show()        





def main(args):
    search(args[0], args[1:])

if __name__ == '__main__':
    main(sys.argv[1:])