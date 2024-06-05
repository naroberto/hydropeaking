# hydropeaking

# This is an image processing pipeline which prepares raw images of 2 cameras for fish tracking in 'TRex'.

STEP 1:
This code is designed to stitch together TIFF images along central area of overlap, ensuring that the two images taken at the same time align correctly and can be merged to one image.  
Specifically, it creates a transformation matrix M for stitching images of an experiment, assuming that the two cameras did not move their orientation. This inconsecutively loaded in step 2.
The transformation matrix M is created for one pair of images and then used for other images of the same experiment in step 2.


STEP 5: 
The initial background subtraction method failed because some fish remained stationary for extended periods, causing them to be included in the background. 
This issue primarily occurred during the ramping phases (up_i, d_i), where short time windows were used to build a median background. 
However, these short time windows were necessary due to the changing water levels and the associated displacement of background objects caused by refraction.
