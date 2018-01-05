**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[01_hog]: ./examples/01_hog.png
[01_hog_notcar]: ./examples/01_hog_notcar.png
[03_classifier]: ./examples/03_classifier.png
[04_sliding_128]: ./examples/04_sliding_128.png
[04_sliding_96]: ./examples/04_sliding_96.png
[04_sliding_64]: ./examples/04_sliding_64.png
[04_sliding_hot]: ./examples/04_sliding_hot.png
[04_heatmap]: ./examples/04_heatmap.png


[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The HOG feature extraction code is in the function `get_hog_features` in `gradient.py`. 

`gradient.py` can be used to produce a plot including a sample image, HOG visualization, and the associated feature vector. An example for a car image:

![01_hog][01_hog]

and a non-car image:

![01_hog_notcar][01_hog_notcar]


#### 2. Explain how you settled on your final choice of HOG parameters.

HOG parameters were selected based on a multidimensional iterative search using a randomized sample of 500 images. You can view the final version of
this code in the function `test_hog` in `classify.py`.

The main dimensions considered were:

   Color Space
   Pixels per Cell
   Cells per Block
   Channels Used

A file showing performance of each variant can be viewed in [hog_performance.txt](./hog_performance.txt). An excerpt:

```
# Count = 500
# colorspace, orient, pix_per_cell, cell_per_block, hog_channel

RGB 9 4 2 0 0.79
RGB 9 4 2 1 0.81
RGB 9 4 2 2 0.77
RGB 9 4 2 A 0.84
RGB 9 4 4 0 0.85
RGB 9 4 4 1 0.79
RGB 9 4 4 2 0.78
RGB 9 4 4 A 0.83
RGB 9 8 2 0 0.82
RGB 9 8 2 1 0.88
RGB 9 8 2 2 0.79
RGB 9 8 2 A 0.85
RGB 9 8 4 0 0.79
RGB 9 8 4 1 0.87
RGB 9 8 4 2 0.83
RGB 9 8 4 A 0.85

HSV 9 4 2 0 0.84
HSV 9 4 2 1 0.71
HSV 9 4 2 2 0.83
HSV 9 4 2 A 0.92
HSV 9 4 4 0 0.76
HSV 9 4 4 1 0.79
HSV 9 4 4 2 0.82
HSV 9 4 4 A 0.89
HSV 9 8 2 0 0.84
HSV 9 8 2 1 0.78
HSV 9 8 2 2 0.82
HSV 9 8 2 A 0.82
HSV 9 8 4 0 0.74
HSV 9 8 4 1 0.81
HSV 9 8 4 2 0.86
HSV 9 8 4 A 0.79

LUV 9 4 2 0 0.75
LUV 9 4 2 1 0.93
LUV 9 4 2 2 0.83
LUV 9 4 2 A 0.91
LUV 9 4 4 0 0.84
LUV 9 4 4 1 0.88
LUV 9 4 4 2 0.84
LUV 9 4 4 A 0.96 *
LUV 9 8 2 0 0.83
LUV 9 8 2 1 0.91
LUV 9 8 2 2 0.86
LUV 9 8 2 A 0.92
LUV 9 8 4 0 0.83
LUV 9 8 4 1 0.88
LUV 9 8 4 2 0.89
LUV 9 8 4 A 0.96 *

# Later runs were repeated and timed

YUV 9 4 4 A 0.96 10.70 *
YUV 9 4 4 A 0.94 10.45 *
YUV 9 4 4 A 0.93 10.15 *
YUV 9 4 4 A 0.94 11.18 *
YUV 9 4 4 A 0.93 10.54 *

YUV 9 8 2 A 0.94 2.49
YUV 9 8 2 A 0.89 2.43
YUV 9 8 2 A 0.89 2.47
YUV 9 8 2 A 0.92 2.54
YUV 9 8 2 A 0.9 2.73

...

```

In general, colorspace YUV and LUV seemed to have the best performance, with pixels per cell / cells per block values of (4, 4) and (8, 2) performing well. Tests that
include training time show that the (4, 4) variants seem to take much longer to execute so (8, 2) was typically used.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The training process is handled in the `train.py` file. The main function, `train_with_spec` takes a nested Python dictionary that
is loaded from a JSON file and contains all of the training and feature extraction parameters including:

   - `count` - the number of vehicle and non-vehicle items to sample
   - `color_space` - the colorspace to use for all feature extraction,
   - `spatial` - a dictionary with parameters for the bin_spatial feature generator,
   - `hist` - a dictionary with parameters for the color histogram feature generator, and
   - `hog` - a dictionary with parameters for the HOG feature generator.

The basic training process consists of:

   - Loading a random sample of vehicle and non-vehicle items
   - Loading the image associated with each item and converting to the colorspace specified
   - Generating a per-column scaler for the entire sample using StandardScaler
   - Generating the label vector associated with the sample
   - Creating a training / test split using `train_test_split`
   - Using a default LinearSVC() classifier and fitting using the training set
   - Displaying the classifier performance score based on the test set

At the end of the training process, the classifier and scaler are serialized to
files with the same base name as the specification file so that they can be 
used by the image / video search process. This allows easy switching between
feature extraction parameters and classifiers without having to modify code.

A useful utility for verifying the classifier pipeline is `test_classify.py` which
loads 32 random car images and 32 random non-car images and classifies them
using the specified spec:

![03_classifier][03_classifier]

This visualization lets you quickly see how well the classifier performs.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Basic sliding window search was implemented using the code presented in the class notes, and can be viewed in `search.py`.

The `slide_window` function generates a series of rectangles indicating the 
sub-images that should be searched.

`search_windows` uses the current image, the windows generated by `slide_window`, 
and the classifier, scaler, and feature extraction specification to perform
feature extraction for the each subwindow followed by a prediction using the
scaled features and classifier. It returns a list of all windows that 
the classifier predicts having vehicles.

`draw_boxes` takes the windows and draws them onto a copy of the original image.

I performed some experimentation with the slide_window parameters. In particular,
`y_start_stop` was adjusted to fit the expected area where vehicles were likely
to appear, and `xy_window` was varied to search for vehicles of different sizes. In testing, I found that `xy_window` sizes of (128, 128), (96, 96) and (64, 64)
proved to be useful for selecting vehicles at a variety of distances.

128 x 128:

![][04_sliding_128]

96 x 96:

![][04_sliding_96]

64 x 64:

![][04_sliding_64]

Final grid with car predictions drawn in blue:

![][04_sliding_hot]

A straightforward improvement would be to use different window sizes in different
parts of the image. Areas higher on the image are likely to contain vehicles that are further away, and should be scanned with large, medium and small windows, while areas lower on the image are likely to contain nearby vehicles and could be scanned with only larger windows.

A second implementation `find_cars`, based on the example from the class notes, uses a slightly different method of generating windows and allows reuse of a single HOG histogram for the entire window. The key difference is that the HOG is generated
a single time for the area of interest, and then rectangular subsets of the
HOG are used for feature extraction along with spatial and color histogram
sample of those same subsets. The example was modified to support the 
use of the same classifier, scaler, and feature spec dictionary as the trainer and `search_windows`.

For this technique, the main experimentation that I performed was with the `scale`
parameter which controls how the image should be resized. Scales of [1.0, 1.5, and 2.0] seemed to work well as well as simply [1.0, 2.0].

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The search function was tested on a number of test images from the test_images/
directory to check for performance.

Performance with the raw prediction windows was reasonable but tended to include a false positives, as well as overlapping windows.

To improve this performance, a simple heatmap was implemented as suggested in the
class notes. This can be seen towards the bottom of the main `search` function
in `search.py`, and consists of four main steps.

First, an empty heatmap image is created, and `add_heat` uses the bounding
boxes to increment any pixels that contain predicted vehicles. Pixels that
are contained in multiple bounding boxes are incremented repeatedly and will
have a higher value.

Second, `apply_threshold` is used to zero out pixels below a certain value. This effectively removes pixels that have not been marked by multiple bounding boxes.

Third, `scipy.ndimage.measurements.label` is used to construct a new set of bounding boxes based on the thresholded heatmap. This list now replaces the original bounding boxes.

An example of a search prediction grid and the heatmap that is produced:

![][04_heatmap]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./test_out/project_yuv.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As described above, the main mechanism for reducing false positives and combining bounding boxes was a thresholded heatmap followed by bounding box construction using 
`scipy.ndimage.measurements.label`.

Additionally, the heatmaps were time-averaged before thresholding. A number of different techniques were used, but ultimately a simple equal-weight average of the most recent four frames was used.

Time-averaging allowed a slightly lower heatmap threshold without adding too many false positives. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest immediate issue that I faced was performance; the current version takes about 30 minutes to process project_video.mp4.

A simple approach would be to reduce the size of the video being processed; instead of 1280x720, I attempted to process it at 640x360. This worked with original sliding window search (after adjusting window sizes) but I was unable to get this to work successfully with `search_cars` which uses HOG sub-sampling. I believe that this is mainly because of the hard-coded window / block size which makes it difficult to change the `scale` parameter below 1.

There are also some structural limitations that force a single colorspace for all feature extraction. It's very possible that different feature extraction techniques could work well with different colorspaces - for instance, spatial binning might work fine in grayscale, and color histograms might work better with a different colorspace than HOG.

Finally, the current predictor produces a simple binary classifiation, and the heatmap simply sums those binary predictions for each pixel. Other predictors (such as a Neural Network classifier) might be able to include a confidence value that would produce a wider range of values for the heatmap, allowing the system to better distinguish between false positives and true positives.
