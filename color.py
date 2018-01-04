import cv2
import numpy as np
import matplotlib.pyplot as plt

from loader import *

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
    

def main(args):
    for item in samples(5):
        path = item['path']
        tag = item['tag']
        img = load_image(path)

        rhist, ghist, bhist, bin_centers, hist_features = rgb_hist(img)

        # Plot a figure with all three bar charts
        fig = plt.figure(figsize=(12,3))        
        plt.subplot(141)        
        plt.imshow(img)
        plt.title('%s' % tag)
        plt.subplot(142)
        plt.bar(bin_centers, rhist[0])
        plt.xlim(0, 256)
        plt.title('R Histogram')
        plt.subplot(143)
        plt.bar(bin_centers, ghist[0])
        plt.xlim(0, 256)
        plt.title('G Histogram')
        plt.subplot(144)
        plt.bar(bin_centers, bhist[0])
        plt.xlim(0, 256)
        plt.title('B Histogram')
        plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])