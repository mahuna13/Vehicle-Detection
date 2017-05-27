##Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: hog.png
[image2]: spatial.png
[image3]: boxes.png
[image4]: boxes_color.png
[image5]: heatmaps.png
[image6]: labels.png
[image7]: img_results.png
[video1]: project_video_out.mp4

### Training SVM for Vehicle Detection

The first step to training a SVM for car detection was to determine what will be the feature vectors that would represent each training image. We have settled on a combination of HOG features, spatial features and color histogram, all combined into one large feature vector.

####1. HOG features

HOG feature extraction was done using `skimage.feature.hog` function provided by `skimage` python module. Code calling this function is in `get_hog_features` function in **Creating Features** section of the python notebook.

I have explored different paramaters to pass to the hog function to achieve maximum accuracy on the test set. Test set was obtained by separating out a random portion of car and non-car images provided. The final set of parameters chosen for hog detection was:

``` python
# hog parameters
colorspace = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
```

I decided that extracting hog features on luminosity channel of LUV space had a pretty good testing accuracy and later performed well on the detection, and continued to use that throughout the project. I also found that using 'ALL' channels worked well, but the amount of computation was three-fold, so I decided to settle on using one channel if that worked sufficiently well. Here is an example image of how HOG features look like, extracted with these parameters on an example car image:

![alt text][image1]


####2. Spatial Features

In addition to HOG features, we also append spatial features to our final feature vector. The code is defined in `bin_spatial`. The idea behind spatial features is to resize an image into a much smaller image, and use that as an input feature. That way we improve the performance significantly than if we had decided to use the original image, but hope that 16x16 still gives us enough information to distinguish between cars and non-cars.

Here are picture examples of car and non car images in their original sizes (left), and when scaled to 16x16:

![alt text][image2]

####3. Color Histogram

Finally, I added color histogram to my final feature vector. Color histograms were calculated for each channel of the image and binned into 32 bins. The code for this  is in `color_hist` code in **Creating Features** section of the python notebook.

####4. SVM training

Once I extracted feature vectors from all the provided car and non-car images, I have split the data into training and test data sets. Then, I used the training data set to train a SVM. I used `sklearn` function `grid_search.GridSearchCV` to explore different combination of kernel and C parameter for training. Grid search returned the best peforming parameters to be *{'C': 10, 'kernel': 'rbf'}* , so I used those to train my SVM. Final test accuraccy achieved was ***Test Accuracy of SVC =  0.9944***

All the code for training, SVM parameter exploration, as well as saving and loading of the svm is provided in **Train SVM** section of the python notebook.

###Sliding Window Search

####1. Sliding Window Search

Now when we can identify car images, we need to decide how we're going to search input images for cars. We implemented a function `find_cars` that does several important steps of vehicle detection:

1. It takes in ranges for Y and X axis of the image to limit the area of the image being searched.
2. It calculates hog features for the whole image (i.e. in our case only 'L' channel)
3. It takes in a scaling factor for the size of the sliding window as well as step size, enabling configurable size window as well as configurable distance between the windows
4. Proceeds to slide the window in the search area of the image
5. Extracts hog features for the window by sampling hog feature of the whole image, and then adds spatial and color features to create a feature vector for the sliding window
6. It runs SVM predict on the sliding window
7. Returns window coordinates if car is detected

This function provides most of the meat of the vehicle detection, but it can only run the sliding window operation on a fixed size window. That's why we have to call the function with many different window sizes.

We searched a variety of different window sizes. On the far left and the far right of the image, we focused on searching larger windows, because the cars that are closer and are just entering the frame appear larger on the image. In the middle of the image, we focused on searching small windows, because the cars way in front of us appear small.

Here is visualization of different window sizes I used. It's a bit hard to see the sizes of the windows because of the overlap between them, but I think that the image at least identifies the different regions searched:

![alt text][image3]

And here are different sized boxes colored differently during vehicle detection. It's clearly visible how dark blue and yellow, the smaller windows, are used to detect the cars further out, while red, green and turqoise, larger windows, are used to detect the cars entering the frame.

![alt text][image4]

####2. False positives

To decrease the frequency of false positives we employ the heatmap methodology described in the lectures. We take the boxes found above, and we construct heatmaps based on those boxes. The more boxes cover a certain area of the image, the hotter that area becomes.

![alt text][image5]

After getting heatmaps, we run a thresholding function to eliminate false positives, aka we want to eliminate areas of the image that have low "temperature" level because those corespond to areas that maybe accidentally got misclassified with one of the sliding windows. In other words, we want to keep only the areas that had multiple sliding windows detect positives.

Finally, we run `scipy.ndimage.measurements.label` that helps us label different heat areas as different cars, so that each car gets exactly one box around it. Here is how the heatmap image looks like after we label it. As you can see, each car is represented with a different color. On the first image, there is exactly one car, in the second, there are two.

![alt text][image6]

The code for all of this is in `detect_vehicles_in_image` function in the python notebook.

#### 3. Image pipeline results

Finally, we draw boxes around each labeled car. This is how the result looks like on the sample images above:

![alt text][image7]
---

### Video Implementation

####1. Video Result
Here's a [link to my video result](project_video_out.mp4)


####2. Video Pipeline Details

Video pipeline is rather similar to my image pipeline, but it uses several additional steps to enhance the performance of frame to frame flow.

First, we **only process every other frame of the video**, keeping track of the order number of the frame in a global variable n_frame. For every frame where we don't run SVM search, we reuse the boxes found in the previous frame. This speeds up our video processing by a factor of two.

Next, we **save the heatmap of the previous frame and add it to the next**. This additionally helps us battle false positives in the combination with thresholding, because one-off false detections will likely happen in one frame but not in the next.

Finally, we try to identify and **correlate the boxes from the previous frame and the current frame that represent the same car**. The code that does this is in the function `combine_boxes` in the python notebook. It calculates the center of each box, as well as the span of the box, and it finds boxes from the previous frame that are really close to the boxes from the current frame, and then we combine the two with a decaying factor of 0.8. This results in a smoother flow of the boxes that are less jumpy from frame to frame.

We also added the optional combining of lane detection with vehicle detection to produce the final result.

The code for video processing is in `detect_vehicles_in_video` and Test on video section of the notebook.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There is one thing we could do to both boost the performance and decrease the number of false positives. We could split the search area into area where cars enter the image, on the far left and right, but then focus the rest of the search around the cars we previously detected in the previous image. So instead of searching the whole image from scratch, we only observe the area surrounding the previously detected car. This would decrease the number of sliding windows we need to test, as well as decrease the number of false negatives by only searching areas where it's likely we will find cars.

Tweaking the sizes of the boxes worked well enough for the video given, but it was full of trial and error, with rather tedious debugging. In addition, there is no guarantee that these windows will work well for any other video provided. Is there a better mechanism of finding which window sizes to use?

My pipeline doesn't really cover the area just in front of the car, but possible a better way to deduce whether something is directly in front would be with a distance sensor.

The pipeline will also fail to detect the actual number of cars present when they overlap in some way on the image. This implies that a better frame-to-frame tracking of vehicles could be achieved. We should also be able to predict relative velocities of each car and be able to predict where it will be in the next frame.

Finally, the pipeline is likely to fail in situations where we have decreased visibility, with light reflecting off the car or in the dark, which could potentially be remedied by retraining the SVN with a different data set.


