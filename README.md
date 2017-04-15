## Vehicle Detection

The goals of this project are the following:

* Extract features from a labeled training set of images and train a classifier Linear SVM classifier.
* The following features have been extracted from the image in the project:
  * Histogram of Oriented Gradients (HOG) feature
  * Color Histogram
  * Spacial Bining
* The features extracted are normalized before training the classifier.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_notcar]: ./samples/car_notcar.png
[bounding_box_heatmap]: ./samples/bounding_box_heatmap.png
[hog]: ./samples/hog.png
[output_video]: ./output_video_project.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook named "Histogram of Oriented Gradients (HOG)".
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
![Car and not car sample training images][car_notcar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
I finally settled on 'YCrCb' color space beacuse I saw that it increased the accuracy of the SVM classifier.

I experimented with various parameters for `skimage.hog()` and finally settled on the following parameters:
`color_space='YCrCb'`
`spatial_size=(32, 32)`
`hist_bins=32`
`orient=9`
`pix_per_cell=8`
`cell_per_block=2`
`hog_channel='ALL'`

Below is an example of HOG features for the 3 image channels for the a sample `car` and a sample `notcar` displayed above.
![HOG features][hog]

#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the `Train SVM Classifier` section of the `vehicle_detection.ipynb` notebook, I did the following:
* Extracted color and HOG features in function `extract_image_features()`
* Scaled the extracted features such that they had zero mean and unit variance using `sklearn.preprocessing.StandardScaler()`
* Split the available training data into 80% training samples and 20% test samples using `train_test_split()`
* Trained a linear SVC classifier using the default parameters.

The trained calssifier has an accuracy of 98.2%

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the `fast_process_image()` function of the `vehicle_detection.ipynb` notebook, I used windows of size 64x64, 96x96 and 128x128 to search for cars.
The `find_cars()` shows that I used an overlap of 75% (2 cells out of the 8x8 cells) between the sliding search windows.
In addition, given that cars appear smaller farther away and larget closer to the camera, I limited the small 64x64 search windows between `y=400` and y=`500` pixels. I limited the search window for the 96x96 windows between `y=400` and `y=500` pixels. Similarly, I limited the search window for the 128x128 windows between `y=450` and `y=600` pixels.

I chose 75% overlap because as I experimented with various percentages of overlap, I saw that 75% overlap performed failrly fast while also being fairly accurate in detecting cars that were close to each other.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used the default LinearSVC classifier. Using 
The below image shows the heatmap, the thresholded heatmap and the actual bounding box drawn based on the thresholded heatmap.

![Heatmap and Bounding Box][bounding_box_heatmap]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video_project.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The classifier detected a fairly significant number of false positives. In order to reduce the false positives, I created a heatmap to mark the areas of the image where a car was detected by the calssifier. Each sliding window that detected a car incremented the values in the heatmap. I summed up heatmaps over 25 frames and only accepted a detection as valid if atleast 15 sliding windows among the 25 frames detected a car. Such a thresholding helped significantly minimize the false positive detections.
I used `scipy.ndimage.measurements.label()` on the thresholded heatmap to detect labels. Assuming each of those labels was a car, I drew bounding boxes using those labels. As another filter, I discarded bounding boxes with a area of `< 2500` pixels. The above image shows how this pipeline works.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. The pipeline still detects many false positives. The accuracy of the classifier can be greatly improved with
   * More training data
   * A more even distribution of training images so that images with different lighting conditions and different cars are more evenly represented in the test and training set.
2. In the above pipeline I am using a heatmap to detect cars and reduce false positives. The main drawback of this approach is that even after I have ahigh confidence detection of a car, if the subsequent series of frames fails to detect the car, the heatmap produces a result that may be lower than the threshold in areas where the car is supposed to be. Therefore, the above pipeline fails to track the car. A better approach will be to use Kalman Filter to estimate the position of the car. If the velocity of the car can be determined, the pipeline can also do a better job of searching areas where the car is predicted to be.
3. The thresholds I set required significant amount of tuning. In a real world scenario, the road and lighting conditions can change significantly. Therefore, the thresholds must be dynamically tunable as the road conditions change.


