## Advanced Lane Finding - Writeup
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Camera Calibration

The code for this step is contained in the IPython notebook located in "./examples/Project 4.ipynb". 

I use the [OpenCV](http://opencv.org/) functions findChessboardCorners() and drawChessboardCorners() to automatically find and draw corners in an image of a chessboard pattern.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Then I've created a function called undistort

'

    def cal_undistort(img, objpoints, imgpoints):
    
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)

        # using CV2 Undistort
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        return undist
 '

#### 2. Perspective Transform.

The code for my perspective transform includes a function called `warped()`:

'

    def warped(img, top_right, top_left, botom_right, botom_left):

        #extract image dimensions
        img_size = (img.shape[1], img.shape[0])
        #set source points
        src = np.float32([[top_right],[top_left],[botom_right],[botom_left]])
        #define width and height
        w, h = img.shape[1], img.shape[0]
        #set destination points
        dst = np.float32([[w,0],[0,0],[w,h],[0,h]])
        # get a perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # get inverse matrix
        Minv = cv2.getPerspectiveTransform(dst, src)
        # warp original image
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        #return warped image and inverse matrix
        return warped, Minv
 '

The `warped()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the source and destination points in the following manner:

top_right = (725, 455)
top_left = (555, 455)
botom_right = (1280, 680)
botom_left = (0, 680)

# Source
'src = np.float32([[top_right],[top_left],[botom_right],[botom_left]])'

# Destination
'dst = np.float32([[w,0],[0,0],[w,h],[0,h]])'

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 725 , 455     | 1080, 0       | 
| 555 , 455     | 0   , 0       |
| 1280, 680     | 1082, 720     |
| 0   , 680     | 0   , 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 3. 

I used a combination of color and gradient thresholds to generate a binary image.  Here's an example of my output for this step.  

![alt text][image3]

I Created a function to combine the best color and gradient thresholds:

'

    def color_and_gradient_threshold(img):
    
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]

        # Grayscale image
        # NOTE: we already saw that standard grayscaling lost color information for the lane lines
        # Explore gradients in other colors spaces / color channels to see what might work better
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        thresh_min = 20
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Threshold color channel
        s_thresh_min = 170
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary

'

#### 4. Binary Function:

To consolidade all the step above I've created a function that gives us a Undistorced Warped Binary image

'
       
    def binary(img):
    
        global objpoints
        global imgpoints
        global top_right
        global top_left
        global botom_right
        global botom_left

        #Correcting image
        undistorced = cal_undistort(img, objpoints, imgpoints)

        #Aplying Thresholds
        color_and_gradient = color_and_gradient_threshold(undistorced)

        #Perspective Transform
        result, Minv = warped(color_and_gradient, top_right, top_left, botom_right, botom_left)

        return result, Minv
'

#### 4. Identified lane-line pixels and fit their positions with a polynomial

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
