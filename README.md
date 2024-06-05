# hydropeaking

### Overview
# This image processing pipeline prepares raw images from 2 cameras for fish tracking in 'TRex'.

STEP 1  (Stitching Images):
Here we stitch together TIFF images using a central area of overlap, ensuring that the two images taken at the same time align correctly and can be merged into one image.  
Specifically, we create a transformation matrix M for stitching images of an experiment, assuming that the two cameras did not move their orientation.
The transformation matrix M is created for one pair of images and then used for other images of the same experiment in step 3.

STEP 2 (Rotation and Cropping): 
Set the upper left corner of the illuminated arena as the origin for rotation. Select five or more points along the horizontal shoreline, with the downstream direction going to the right. 
Calculate the average rotation angle based on these points. Rotate the image using this average rotation angle and crop the image to an identical size for all experiments. 
The center of rotation remains the same for all experiments (x=1850, y=46), ensuring identical positioning across images and experiments.

STEP 3 (Stitching and Rotating pipeline):
Goal:   Pipeline to stitch and rotate images of one Experiment based on prederived params (from step_1 and step_2)
Input:  raw .tif image streams from two cams that were recording instantaneously (i.e. synchronized in time and at framerate of 15 fps)
Output: stitched, rotated, cropped .tif stack for each experiment

STEP 4:
Plot hydraulic data. Here we ensured temporal synchronization, between hydraulic data and camera recordings. 

STEP 5 (Background Subtraction and Video Creation):
Subtract the median background for base and peak flows and store the results. 
Subdivide the sequence of pre-processed images into .mp4 videos for experimental phases: acclim, up_1, p_1, d_1, b_1, up_2, p_2, d_2, b_1, up_3, p_3, d_3, b_3. 
Check the timing of phases and run it for b_2 and all experiments.

STEP 5b (step_5_up_down.py) (Alternative Background Subtraction): 
If the initial background subtraction method failed because some fish remained stationary for extended periods, causing them to be included in the background. 
This issue primarily occurred during the ramping phases (up_i, d_i), where short time windows were used to build a median background. 
However, these short time windows were necessary due to the changing water levels and the associated displacement of background objects caused by refraction.
