#Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_undist_example.png "Undistorted"
[image2]: ./output_images/wrap_example.png "Warp Example"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 8 through 41 of the file called `p4.py` (Class `Undistorter`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Undistorted test images can be found in `output_images` folder (`test[1-6]_undist.jpg`)

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 65 through 85 in `p4.py`). I convert an image to HLS, than take pixels where S channel is between (120, 255) and H channel is between (15, 100) and combine them with gradient in x direction applied to L channel with threshold (25, 100). Images from this step are in `output_images` folder (`test[1-6]_binary.jpg`)


####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in a class `Warper`, which appears in lines 44 through 62.  It takes as inputs source (`src`) and destination (`dst`) points.  Its functions take an image (`img`) as an input. I chose the hardcode the source and destination points in the following manner (lines 311-320):

```
src_trans = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 35, img_size[1]],
    [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
dst_trans = np.float32(
    [[(img_size[0] / 5), 0],
    [(img_size[0] / 5), img_size[1]],
    [(img_size[0] * 4 / 5), img_size[1]],
    [(img_size[0] * 4 / 5), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 256, 0        | 
| 203, 720      | 256, 720      |
| 1102, 720     | 1024, 720     |
| 705, 460      | 1024, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image2]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find lanes I first compute a histogram along all the columns in the lower half of the image and find the biggest peak on the right and left parts of an image. These points serve as a starting points for searching lanes (`find_lane_start` function, lines 88-97).
Then I break the image in layers and starting from the bottom layer I do the following:

 1. Find all pixels around 100 pixel margin around the starting point
 2.  Add coordinates of these pixels to an array
 3. If number of found pixels is more then some threshold (minpix = 550) then lower the margin for the next layer to 50 pixels and set the starting point to the mean of x-value of the pixels, Otherwise the starting point is unchanged and margin is set to 100.

In the end I have two arrays with coordinates for left and right lines which I use to fit 2nd order polynomials to find left and right curves.
(`find_lanes` function, lines 99-168)
 
Images from this step are in `output_images` folder (`test[1-6]_lanes.jpg`)

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the radius of curvature I scaled coordinates of lines pixels and used them to fit new polynomials in world space. Then using formula  $R_{curve} = \frac{(1+(2Ay+B)^2)^\frac{3}{2}}{|2A|}$ (where A and B are coefficients of polynomials) I calculated ROCs for left and right lines and took the mean of them for the final ROC.

To calculate the position of the vehicle with respect to center I first calculated the bottom points of the left and right line. Then assuming that the width of the lane is 3.7m and that the camera is mounted at the center of the car I calculated the offset of the car from the center.

I did all this in the `write_roc_and_offset_on_img` function, lines 221 through 241

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the `draw_poly_on_img` function (lines 196 through 219) and `write_roc_and_offset_on_img` function (see above). Images from this step are in `output_images` folder (`test[1-6]_final.jpg`)

---

###Pipeline (video)
The whole pipeline logic is gathered in the `process_frame` function (lines 262-303). For video I compute lane lines in the following way. I try to fit polynomials then check them with the `check_lanes` function (lines 173-194).  In this function I search for the min and max distances between the left and write polynomials and if the difference between min and max is more then some threshold (200 pixels) I consider the polynomials bad.  
Then If I did not find lines or they were bad I use lines from the previous frame. Otherwise I smooth the lines between frames by applying exponential moving average with alpha=0.2.
Also if I found good lines on the previous frame I use bottom points of them as the starting points for searching line pixels on the next frame
  
####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a link to my video result: https://youtu.be/RVoE6uVy0GQ

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I tried to process the `challenge_video.mp4` but it appeared that yellow lines in it almost is not detected by my thresholding function also for me they look the same as in the `project_video.mp4`. Also I found out that after the perspective transform lines are not parallel and a different transformation parameters are needed. It was mentioned in lectures that there are algorithms for detecting four source points in an image programmatically but unfortunately there was no examples.
