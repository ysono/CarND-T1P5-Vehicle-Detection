**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

The main submissions are these two python notebooks:

- [project-train.ipynb](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-train.html)
- [project-test.ipynb](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-test.html)

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Parameters for HOG are as suggested in the coursework:

- 8 pixels per cell
- 2 cells per block
- 9 orientations

These values were visually confirmed to be sufficient in resolution and contrast. With the subsequent training of an SVM, they are demonstrated to be sufficient as well. These were also validated in the context of the eventual window search; using 2 cells per step is shown to adequately cover areas of interest.

The implementation of extraction of HOG features during training is defined in [`hog_helper` and `create_hog_features_getter` in the `train` notebook](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-train.html#Extract-hog-features-only) and that during testing and predicting is defined in [`get_hog` in the `test` notebook](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-test.html#Adjust-feature-extraction-based-on-which-classifier-model-was-chosen-above).

#### 2. Explain how you settled on your final choice of HOG parameters.

The evolution of choice of features was as follows:

1. Collect HOG features only, because this was the only known method of detecting shape. Start with a color space that encodes non-hue information in one channel, such as YUV. The code starts [here](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-train.html#Extract-hog-features-only).
1. Use a color space that encodes non-hue information in two channels, namely HSV and HLS.
1. Collect HOG, color spatial and color histogram features. For color features, start with the color space that yielded the highest accuracy with HOG features only: HSV. The code starts [here](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-train.html#Extract-hog-and-color-features).
1. Under the assumption that red color features were important because red taillights were always expected to be present, red channel entered consideration.
1. When shrinking images in order to extract color features, different target resolutions were tried.
1. So far the choice of color channels had been guided by classifier accuracy, i.e. without visualization. However, the best color channel, HSV could not deteect black cars. A search for an optimal color space was re-initiated using code samples provided in lecture and by testing the classifier on test images and videos. In the end, L and S channels from HLS space were used for HOG and color features, and R channel was used for color features.

The use of `sklearn.preprocessing.StandardScaler` is warranted even if not using color features, because we want to normalize them to zeor mean and unit variance.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The original, now legacy, pipeline for training was as follows:

1. Collect some features from the small dataset of cars and noncars.
1. Optimize a `sklearn.svm.SVC` classifier using `sklearn.model_selection.RandomizedSearchCV`.
1. For reference, train a separate classifier anew, using the optimized parameters as guidance. For training data, collect features from the large dataset of cars and noncars.

The pipeline would then be repeated for a different feature set, chosen as described in the above section.

However, over time, it was observed that the optimized kernel would usually be `linear`, so  `sklearn.svm.LinearSVC` was always used as the reference classifier. Then, it was observed that `sklearn.svm.LinearSVC` would always train faster and yielded a more accurate model. Therefore, the optimization step was removed, and the pipeline became simplified as follows:

1. Collect features from the large dataset of cars and noncars.
1. Train a `sklearn.svm.LinearSVC`, using the classifier's default parameters.

This pipeline was also repeated for different feature sets.

The implementation of the pipelines is [here](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-train.html#Pipeline-functions). The way to swap in different strategies of feature collection is to reassign the global function `GET_RAW_FEATURES_FROM_IMAGE`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

It was first decided that window search would happen in distinct strips of an input image, such that the strips would each fit one window vertically. Not only does it simplify the math a little, it asserts that from the camera's perspective, all objects at the same zoom are aligned horizontally in an image. Actually, this is not strictly true unless the image has its camera distortion corrected and unwarping applied, as done in the previous Advanced Lane Finding project, but for the purpose of vehicle detection, the assertion should generally hold.

The vertical position and height of the strips, as well as the number of them, were arbitrarily chosen, after visualizing them on test images. The final strips work for the 3 cars that appear over the two videos. Note, there are no tall or long vehicles present in the videos.

The implementation and visualization are [here](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-test.html#Define-strips-in-which-to-search-for-windows).

Windows within each strip is determined in the following way:

1. First, the strip is vertically scaled to 64 pixels, because HOG features are derived from 64 x 64 subimage and we want one window in the strip vertically. Horizontal scaling is proportional, but some adjustment is added such that the rightmost stepped window touches the right edge. The math is `w = round((strip_rgb.shape[1] / strip_rgb.shape[0] * pixels_per_window) / pixels_per_step) * pixels_per_step`.
1. The number of windows is derived after running `skimage.feature.hog`. The positions of windows in pixels are not calculated at this time.
1. Extracting color features requires a different scaling, to height of 16 pixels. This time, the width is calculated from the number of windows: `w = pixels_per_step * (num_windows_x - 1) + h`. We know the exact pixel locations of windows in the 16-high dimension (x boundaries of first window are 0 and 15, second are 4 and 19, etc); and we map these back to the strip's original 1280-wide dimension, as done in this line `xs_unresized = (np.array([x0, x1]) / resize_ratio).astype(np.int)`.

You will notice in the visualization that the windows are not rectangular. This is expected because of resizing. Calling `get_window_features_in_strip` or `select_windows` with `verbose=True` turns on debugging printing, which ensures resizing is calculated correctly. (To visualize where all windows are situated, you can follow this line `predictions = CLF.predict(scaled_features)` with `predictions = np.ones_like(predictions)`, and run `annotate_selected_windows_on_test_image_verbose`.)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The implementation and visualization of the pipeline is [here](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-test.html#Run-prediction).

As mentioned above, I used the test images, which are video screenshots, to choose the optimal feature set extraction strategy, accompanied by the SVM that takes this feature set.

This makes sense from the traditional training/validation/test dataset split. We have these 3 datasets:

1. training portion split from GTI and KITTI dataset (the `vehicles.zip` and `non-vehicles.zip`)
1. validation portion split from the same GTI and KITTI dataset
1. video screenshots in [./test_images/](test_images), and the videos themselves, [test_video.mp4](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/test_video.mp4) and [project_video.mp4](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project_video.mp4)

The roles of the above datasets are:

1. train internal parameters of a single `LinearSVC`
1. compare different combos of feature set extraction strategy and its accompanying classifier
1. (same as above)

One problem is that the number of training samples is looking somewhat small compared to the number of features. Whereas this is probably an issue for a neural net, I'm not sure if it is for SVM. We have `7*7*2*2*9*2 + (16*16+32)*3 = 4392` features trained on `17760 * 0.8 = 14208` training (excluding validation) samples, at a ratio of `14208 / 4392 = 3.23`. The `4392` and `17760` figures can be verified in code [here](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-train.html#Try-hls-for-color).

For this reason if I were to do this project again, I would use udacity's huge dataset, and mix it into some or all of the 3 datasets.

As for of what I did to improve performance, in terms of accuracy of desired results on the videos, it required a lot of try and error of color space usages. Intuition about the importance of saturation and red color spaces didn't really help.

And in terms of time performance, results in [project-train.html](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-train.html) do show that using more color channels adds time to feature extraction. Color transformation adds time, but we almost definitely need some color space other than RGB/BGR anyway. Resizing does not seem to cost too much time, so for color features, it is probably advantageous to use smaller resolution. My final choice of feature set ends up using the most number of color spaces among all experimented -- 2 for HOG and 3 for color. But the performance for video is adequate, as explained below.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

My outputs are:

- [annotated_test_video.mp4](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/output_images/annotated_test_video.mp4)
- [annotated_project_video.mp4](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/output_images/annotated_project_video.mp4)
- Annotated on top of advanced lane line detection: [annotated_project_video_with_lane_lines.mp4](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/output_images/annotated_project_video_with_lane_lines.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The implementation and visualization are [here](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-test.html#Pass-found-windows-to-heatmap).

1. The heatmap for the current frame is created in function `get_heatmap_for_one_frame`.
1. The current heatmap it is summed pixel-wise with heatmaps of the 3 most recent frames. Thresholding is applied on the summed heatmap, such that in at least one frame, the area is an overlap of 2 or more windows. NOTE, this thresholding is probably too restrictive, but it does eliminate most false positives. This happens in function `apply_threshold`.
1. The thresholded summed heatmap is interpreted into boxes of cars using `scipy.ndimage.measurements.label` function. This happens in my function `get_label_boxes_from_heatmap`.

During visualization, the history of recent frames is mocked to be an empty list, but during annotation of videos, a history of 3 frames is remembered. See `heatmap_history` variable in [here](https://ysono.github.io/CarND-T1P5-Vehicle-Detection/project-test.html#Video).

The pipeline runs somewhat real-time, at 0.16 sec per frame ~= 6 frames per sec.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Issues encountered mainly had to do with my approach. I should have spent more time on refining and visualizing feature set extraction, before spending time on improving classifier.

The black color is obviously an issue. In fact I don't know how my pipeline performs on other colors, because there are only 2 black cars and one white car in the two videos. To correct this, I'd need to add more features, which could be reasonable if training with the much larger udacity dataset, or fine-tune choice of color spaces further.

The commonality of false positives is another issue. Many of them appeared to contain a lane line. To remedy this, hard negative mining should be done by manually cropping out false positive windows and augmenting the training dataset with them. And if non-car training samples are under-represented, it could be sufficient to augment them with any road-related, non-car images, such as a collection of traffic signs from the previous traffic sign recognition project.

Even when a car is generally correctly identified, the bounding box is often too short, because I have only 3 horizontal strips in which I search for cars. More strips need to be added.

My current thresholding is primarily aimed to reduce false positives. This also causes some false negatives, and in case where a car box disappears and reappears immediately, some smoothing technique should be able to fill in these holes. This smoothing would use historical positions to derive relative speed and estimate the box's position at the next frame. So that such smoothed-in, non-detected car boxes are not kept around forever, they could be eliminated once their height diminshes beyond some threshold.

The current classifier would have difficulty recognizing vehicles that appear different (such as non-passenger, military, and sports cars), and/or with large feature-less surfaces, and/or are tall or wide, and/or are nearby the camera, because the classifier is not trained for such features or perspectives. However, tall vehicles that are reasonably far away could be accommodated with little or no change to the strip positions.
