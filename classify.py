import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

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
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
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

def search_windows(img, windows, clf, scaler,
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))        

        #4) Extract features for that window using single_img_features()


        features = single_img_features(test_img,
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)        

        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            # fig = plt.figure(figsize=(12,4))
            # plt.subplot(131)
            # plt.imshow(test_img)
            # plt.title('Original Image')
            # plt.subplot(132)
            # plt.plot(features)
            # plt.title('Raw Features')
            # plt.subplot(133)
            # plt.plot(test_features[0])
            # plt.title('Normalized Features')
            # fig.tight_layout()
            # plt.show()

            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


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

    for colorspace in ['LUV', 'YUV']:
        for pix_per_cell in [4, 8]:
            for cell_per_block in [2, 4]:
                for hog_channel in ['ALL']:
                    for i in range(5):
                        items = samples(count)
                        for item in items:
                            load(item)
                        t = time.time()
                        features = scale_features(extract_hog_features(items, colorspace, orient, pix_per_cell, cell_per_block, hog_channel))
                        labels = item_labels(items)

                        rand_state = np.random.randint(0, 100)
                        X_train, X_test, y_train, y_test = train_test_split(
                            features, labels, test_size=0.2, random_state=rand_state)

                        svc = LinearSVC()
                        svc.fit(X_train, y_train)
                        t2 = time.time()
                        score = svc.score(X_test, y_test)

                        dt = t2 - t
                        
                        print(colorspace, orient, pix_per_cell, cell_per_block, hog_channel, score, '%2.2f' % dt)

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel,
    spatial_feat=True, hist_feat=True, hog_feat=True
):
    
    draw_img = np.copy(img)
   
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = img_tosearch

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image

    if hog_channel == 'A' or hog_channel == 'ALL':    
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    elif hog_channel == 0:
        hog1 =  get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    elif hog_channel == 1:
        hog2 =  get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    elif hog_channel == 2:
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
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
            if spatial_feat:
                spatial_features = bin_spatial(subimg, size=spatial_size)            
                features.append(spatial_features)

            if hist_feat:
                hist_features = color_hist(subimg, nbins=hist_bins)
                features.append(hist_features)


            if hog_feat:
                # Extract HOG for this patch
                if hog_channel == 'A' or hog_channel == 'ALL':                        
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                elif hog_channel == 0:
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = hog_feat1
                elif hog_channel == 1:
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = hog_feat2
                elif hog_channel == 2:
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = hog_feat3
                features.append(hog_features)


            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack(features).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return boxes

def test_search(args):
    count = 1000

    color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 16 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [500, 700] # Min and max in y to search in slide_window()    

    cars = vehicles()[:count]
    notcars = non_vehicles()[:count]
    
    for c in cars:
        load(c)

    for nc in notcars:
        load(nc)

    for i in cars[:5]:
        img = load(i)
        print(img.shape, img.dtype)

    car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                      

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)    

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print(X_train.shape, X_train.dtype)

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    # Train the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    ## Test classifier
    if False:
        fig, axs = plt.subplots(8, 8, figsize=(12, 12))
        fig.subplots_adjust(hspace = .2, wspace=.001)
        axs = axs.ravel()


        scaler = X_scaler

        for i in np.arange(32):
            item = random.choice(cars)
            img = load(item)

            features = single_img_features(to_colorspace(img, color_space), 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)        

            #5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))                
            # print(test_features.shape, test_features.min(), test_features.max())        
            #6) Predict using your classifier
            prediction = svc.predict(test_features)
            if prediction == 1:
                axs[i].set_title('car', fontsize=10)
            else:
                axs[i].set_title('not-car', fontsize=10)

            axs[i].axis('off')
            
            axs[i].imshow(img)

            for i in np.arange(32, 64):
                item = random.choice(notcars)
                img = load(item)

                features = single_img_features(to_colorspace(img, color_space), 
                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                    orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat, hog_feat=hog_feat)        

                #5) Scale extracted features to be fed to classifier
                test_features = scaler.transform(np.array(features).reshape(1, -1))                
                # print(test_features.shape, test_features.min(), test_features.max())        
                #6) Predict using your classifier
                prediction = svc.predict(test_features)
                if prediction == 1:
                    axs[i].set_title('car', fontsize=10)
                else:
                    axs[i].set_title('not-car', fontsize=10)
                
                axs[i].axis('off')
                axs[i].imshow(img)        

        plt.show()

        return


    for path in args:
        image = load_image(path)
        draw_image = np.copy(image)
        feature_image = to_colorspace(image, color_space)

        if False:            
            windows = slide_window(feature_image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                                xy_window=(96, 96), xy_overlap=(0.5, 0.5))

            hot_windows = search_windows(feature_image, windows, svc, X_scaler,
                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                    orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat, hog_feat=hog_feat)                       

            out_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
        else:
            ystart, ystop = y_start_stop
            all_windows = []
            for scale in [1.0, 1.5]:
                hot_windows = find_cars(feature_image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel,
                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat        
                )
                all_windows.extend(hot_windows)
            out_img = draw_boxes(draw_image, all_windows, color=(0, 0, 255), thick=6)
        plt.imshow(out_img)
        plt.show()



def main(args):
    test_search(args)

if __name__ == '__main__':
    main(sys.argv[1:])