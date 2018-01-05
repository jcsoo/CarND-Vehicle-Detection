import cv2
import numpy as np
import matplotlib.pyplot as plt

from loader import *

COLOR_SPACES = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']

def to_colorspace(img, color_space='RGB'):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            raise Exception('Unknown colorspace %s' % color_space)
    else: 
        feature_image = np.copy(img)         
    return feature_image    

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to compute color histogram features  
def rgb_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features
    
# Generate spatial vector
def bin_spatial(img, size=(32, 32)):    
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, tuple(size)).ravel() 
    # Return the feature vector
    return features

def plot_histogram(item, color_space='RGB'):
    path = item['path']
    tag = item['tag']
    img = to_colorspace(load_image(path), color_space)

    rhist, ghist, bhist, bin_centers, hist_features = rgb_hist(img)

    # Plot a figure with all three bar charts
    fig = plt.figure(figsize=(12,3))        
    plt.subplot(141)        
    plt.imshow(img)
    plt.title('%s' % tag)
    plt.subplot(142)
    plt.bar(bin_centers, rhist[0])
    plt.xlim(0, 256)
    plt.title('%s Histogram' % color_space[0])
    plt.subplot(143)
    plt.bar(bin_centers, ghist[0])
    plt.xlim(0, 256)
    plt.title('%s Histogram' % color_space[1])
    plt.subplot(144)
    plt.bar(bin_centers, bhist[0])
    plt.xlim(0, 256)
    plt.title('%s Histogram' % color_space[2])
    plt.show()

def plot_bin_spatial(item):
    path = item['path']
    tag = item['tag']
    img = load_image(path)

    fig = plt.figure(figsize=(24,3))
    plt.subplot(171)
    plt.imshow(img)
    plt.title('%s' % tag)    

    sp = 172

    for cs in COLOR_SPACES:
        feature_vec = bin_spatial(to_colorspace(img, cs), size=(32, 32))
        plt.subplot(sp)
        plt.plot(feature_vec)
        plt.title(cs)
        sp += 1
    plt.show()

def main(args):
    for item in samples(5):
        plot_bin_spatial(item)
        # plot_histogram(item, args[0])


if __name__ == '__main__':
    main(sys.argv[1:])