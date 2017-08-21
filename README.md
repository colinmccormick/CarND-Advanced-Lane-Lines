# Advanced Lane Finding Using OpenCV

This project is part of the [Udacity Self-Driving Car Nanodegree Program](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project). The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The main pipeline is in [advanced_lane_finder.ipynb](advanced_lane_finder.ipynb).

## 1. Camera calibration and undistortion

The physical imperfections in cameras lead to distortions in the images that they capture. I need to correct these distortions through a camera-specific calibration procedure. This procedure determines the coefficients used in polynomial expressions that correct radial and transverse distortions. I also need to determine the focal length(s) and lens offset(s) of the specific camera that captured the images I will use. A detailed description of this is [here](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html).

The code for the calibration is in calibrateCamera(). Using the provided chessboard images, I find the chessboard corners, draw them on the image, and then refine them using cv2.cornerSubPix(). This function uses the fact that the gradient at every point near the corner is orthogonal to the vector between the corner and that point; iteratively minimizing the dot product of the gradient and corner-to-point vector refines the corner position.

One complication is that I don't know the real-world coordinates of the chessboard corners, relative to the camera. But I can handle this by shifting our reference frame to use the chessboard itself as the origin (0,0,0) of the real-world coordinates. I can also use the size of a chessboard square as the unit distance in the real world, so the corner coordinates are now really simple: (1,0,0),(2,0,0),(1,1,0),etc. This seems like a trick, but it's just using smart rescaling of coordinates. 

![Chessboard corner finding](./pipeline_examples/chessboard_corners.png)

Next I take the list of image points and object points and use cv2.calibrateCamera() to determine the camera matrix (which specifies the intrinsic camera features: focal lengths and lens offsets) and the coefficients for the distortion-correction polynomials. Using this matrix and coefficients, it's straightforward to undistort new images from this camera using cv2.undistort(). Below: original image (left), undistorted (right).

![Original and undistorted chessboard](./pipeline_examples/chessboard_undistort.png)
![Original and undistorted image](./pipeline_examples/original_undistorted.png)

## 2. Creating a thresholded binary image of lane pixels

This is the most challenging part of the project, and the code is in selectLanePixels(). Following the course discussion and feedback from reviewers, I decided to use the LAB color space for yellow (the B channel) and the LUV color space for white (the L channel). After playing around with different values for the ranges, I settled on values for each (190,255 for B and 225,255 for L). Based on some discussion from other students, I also chose to normalize the B channel to maximum brightness (255), unless the maximum pre-normalization value was below a certain threshold, which helps mask out false positive detections. I ultimately decided not to use Sobel gradients. The union of all pixels that passed at least one test is then returned as a binary (one-channel) image. 

![Lane pixel detection](./pipeline_examples/lane_pixel_detection.png)

## 3. Calculating the perspective transform and warping image

To transform the image perspective to a birds-eye view, I first use two test images of straight road segments to calculate the vanishing point for this camera view with findVanishingPoint(). I do this with techniques from the first project: a gaussian blur to smooth the image; a Canny edge detector to find the lines in the ROI; and a Hough transform to find the dominant lines. Once I have the lines, I have to calculate where they intersect, which is the vp. There may be more than two of them, so I use a linear algebra technique: rearrange the slope and intercept to form a matrix, and solve using numpy's least squares function.

![Vanishing point determination](./pipeline_examples/vanishing_point.png)

Once I have the vanishing point, it's easy to determine the four corner points of the ROI; I set vertical limits on the ROI and calculate the horizontal positions that fall on the vanishing lines. Using those four points and the four corners of the ROI, I use cv2.getPerspectiveTransform() to determine the warping matrix (within the function calculatePerspectiveTransform() ). I can then apply it easily using cv2.warpPerspective(). Below: image with ROI before transform (left); warped image (center); image after warping back (right).

![ROI before and after perspective transform](./pipeline_examples/roi_before_after_warp.png)

## 4. Detect lane pixels and fit lane lines

To detect the lanes, I warp the binary image of lane pixels to the birds-eye view, using the perspective transform I just calculated. I then use the sliding-window technique. To do this intuitively, I use a Lane class, defined at the beginning of the notebook, and create two instances for the left and right lanes. The main lane line finding routine is findAndFit(), which runs in two different modes depending on whether I found a valid fit for the lane in the previous frame. If not (e.g. the first frame) I search the lower quadrant of the image for the starting X position of the lane (left or right), and then look for a centroid of pixels in sliding windows moving vertically. The center of those windows is updated based on the centroid found in the one immediately lower. 

For cases where the previous frame found a fit, the search is confined to windows that are positioned based on the fit, so I'm only searching in a small window around where the lane in the previous frame was positioned. Once the centroids are found, the horizontal and vertical positions of the lane in these windows are then fit using a second-order polynomial. Finally, I apply a sanity-check test to see if I should accepth the proposed fit, which is based on whether the left and right fits at the bottom of the screen are the expected distance apart (the lane width).

![Sliding windows](./pipeline_examples/sliding_windows.jpg)

## 5. Draw lane on original image and calculate curvature and lane offset 

I save the lane fits from the last 4 frames, and use this to compute and draw the lane position in the warped space, in the function drawLane(). Next I warp back to the original view (using the inverse perspective transform calculated above) and impose this lane marker on the frame. 

To better visualize the lane pixel detection and lane marker fit, I write a small (40% scale) version of the sliding window fit into the upper-right corner of the frame.

I calculate the lane offset by simply determining the lane center from the position of the two lane lines, and determine how far off center in the image this is. Using the provided conversion from pixels to meters, I can calculate the true offset. This is under 30 cm in most frames, which makes sense. Calculating the lane curvature is slightly more difficult. Using the  formula for curvature from a second-order polynomial fit, I have to accurately include the x and y pixel calibration in the fit formula. The algebra for doing that is encapsulated in the Lane class function calculateCurvature(), which also averages the curvature of previous frames. I write the offset and curvature directly onto the frame using cv2.putText().

As an extra piece of analysis, I use the intermitent white lane markers on the right side, which are a uniform 40 feet/16 meters apart, to measure the speed of the vehicle. I do this by testing the bottom-most sliding window to see if there are pixels stretching the full width from top to bottom - this indicates that a lane marker is present, since they are longer than the sliding window is tall. By tracking how many frames it takes until the next lane marker appears in this same window, I can then calculate the vehicle's speed (this also requires knowing the frame rate of the video, which I extract using clip.fps). I convert this to km/h and write it on the image. The result is about 110-120 km/h (68-74 mph), which is reasonable.

![Lane marker](./pipeline_examples/lane_marker.png)

The video of the entire pipeline applied to the project video is [here](./output_video/video_output.mp4). The pipeline performs reasonably well in the project video, although it struggles a little with the shadow regions. This isn't too surprising, since it's difficult to distinguish the white lines in the shadow, and the L channel isn't very helpful. The frame averaging technique helps somewhat. 

## 6. Ideas

1. Improving centroid finding in the sliding window technique. Right now the pipeline integrates the window pixels vertically (a histogram) and then identifies all non-zero positions with np.flatnonzero(). Taking the mean of this gives the mean index position of these non-zero vertical integrations of pixels, a generally effective technique. It tends to fail when the pixel detection has mostly picked up edges, or when there are any stray pixels around. This could be improved by thresholding the histogram to drop any horizontal positions that have too few (but not zero) pixels.

2. Implementing shadow-based detection. The lane pixel identification method struggles in shadow. There are algorithms to identify shadowed pixels, based on the physical property that shadows are bluer than sunlight pixels (due to Rayleigh scattering) - see [this paper](https://asp-eurasipjournals.springeropen.com/articles/10.1186/1687-6180-2012-141). It might be possible to first identify pixels that are in shadow in the raw (undistorted) images, and then apply a different lane-pixel-finding algorithm to them, possibly just by using different threshold values in HSL or LAB color space. 

3. Better sanity checks and look-ahead. Since lanes can't actually change curvature faster than a certain amount, it should be possible to threshold between frames on this amount and discard fits that are obviously outside of this tolerance. As for look-ahead, there is information in the image very close to the vanishing point that we're not currently using, because it's just outside of the ROI. It might be possible to divide the field of view into two vertical slices, and calculate fits and curvature in them separately. That would give us a second sanity check, because the following frame should find the curvature of the lower ROI changing to be closer to the curvature of the previous frame's upper ROI.
