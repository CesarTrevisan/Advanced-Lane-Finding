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
[image5]: ./writeup_images/fit.png "Fit Visual"
[image6]: ./writeup_images/final.png "Final"
[image7]: ./writeup_images/histogram.png "Binary Warped"
[image8]: ./writeup_images/histogram2.png "Histogram"

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


    def cal_undistort(img, objpoints, imgpoints):
    
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)

        # using CV2 Undistort
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        return undist


#### 2. Perspective Transform.

The code for my perspective transform includes a function called `warped()`:

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

The `warped()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the source and destination points in the following manner:

top_right = (725, 455)
top_left = (555, 455)
botom_right = (1280, 680)
botom_left = (0, 680)

##### Source
    src = np.float32([[top_right],[top_left],[botom_right],[botom_left]])

##### Destination
    dst = np.float32([[w,0],[0,0],[w,h],[0,h]])

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


#### 4. Binary Function:

To consolidade all the step above I've created a function that gives us a Undistorced Warped Binary image
       
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

#### 4. Identified lane-line pixels and fit their positions with a polynomial

After applying calibration, thresholding, and a perspective transform to a road image, I have a binary image where the lane lines stand out clearly. However, you still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

I've used a Histogram to identify where on image is the lanes

![alt text][image7]

With this histogram I am adding up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I can use that as a starting point for where to search for the lines. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

![alt text][image8]

Then I create a function to fit my lane lines with a 2nd order polynomial:

Fitlines Function takes a binary warped images and give us:

* left_fit
* right_fit
* out_img
* lefty
* leftx
* righty
* rightx
* ploty

        def fitlines(binary_warped):
            # Assuming you have created a warped binary image called "binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255


            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(binary_warped.shape[0]/nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Set the width of the windows +/- margin
            margin = 100
            # Set minimum number of pixels found to recenter window
            minpix = 50
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 


            # Fit a second order polynomial to each
            if len(leftx) == 0:
                left_fit =[]
            else:
                left_fit = np.polyfit(lefty, leftx, 2)

            if len(rightx) == 0:
                right_fit =[]
            else:
                right_fit = np.polyfit(righty, rightx, 2)



            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )



            return left_fit, right_fit,out_img, lefty, leftx, righty, rightx, ploty
            
![alt text][image5]
            

#### 5. Radius of curvature of the lane.

To calculate lanes curvature I've used: lefty, leftx, righty, rightx and ploty to feed curvatures function:


    def curvatures(lefty, leftx, righty, rightx, ploty):

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/730 # meters per pixel in x dimension

        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) /                                     np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) /                                   np.absolute(2*right_fit_cr[0])

        return left_curverad, right_curverad


#### 6. Lane Ploted into Image.

To Plot Lane into image and write some usefull information I've created draw_lane function

    def draw_lane(img, warped, left_fit, right_fit, ploty, left_curverad, right_curverad, Minv):

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 

        # Creating Text and set font parameters
        TextL = "Left  Curvature: " + str(int(left_curverad)) + " m"
        TextR = "Right Curvature: " + str(int(right_curverad))+ " m"
        fontScale=1.5
        thickness=3
        fontFace = cv2.FONT_HERSHEY_SIMPLEX

        # Using CV2 putText to write text into images
        cv2.putText(newwarp, TextL, (110,60), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
        cv2.putText(newwarp, TextR, (110,110), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        return result


![alt text][image6]

---

### Pipeline (video)

I've created a Pipeline Function to consolidate all steps above:

    def pipeline(img):

        # creating a Binary Undistorced Warped Image
        binary_warped, Minv = binary(img)

        # Fiting Lines
        left_fit, right_fit,out_img, lefty, leftx, righty, rightx, ploty = fitlines(binary_warped)    

        # Calulating the left and right lines curvatures
        left_curverad, right_curverad = curvatures(lefty, leftx, righty, rightx, ploty)

        # Draw Lane between road lines
        result_lane = draw_lane(img, binary_warped, left_fit, right_fit, ploty, left_curverad, right_curverad, Minv)

        return result_lane    



#### 1. Final Result.

Here's a [link to my video result](https://github.com/CesarTrevisan/Advanced-Lane-Fiding/blob/master/out_test_video.mp4?raw=true)

---

### Discussion

#### 1. Sumary

I used Python CV2 functions to undistort a image using many images from tha same camera, I searched and found conners of a chess table that I used to creat a transformation Matrix to undistort images. Then I focus in a specific road area, to do this a apply a perspective tranformation to gives us a "bird view" of road. So, using Color and Gradient Threshold tecniques I highlighted lanes lines of road creating a Binary image where the points are lane areas. 

To identify where Lanes is I used Slide Windowns, looking for pointo to fit a in a secound degree polinomial funcion and give us the left and right curvature. 

To finish and present the result I plot lane area and information about curvatures on the images, and also create a video. 

#### 1. Considerations

For the Project video, the tecniques described worked very well, to harder and challeng video the pipeline needs some fine tuning or even adition of another techniques to improve the results. In Conclusion, the set of image analysis techniques proved to be a very robust way to identify lane lines in the road. 
