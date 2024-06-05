# hydropeaking

# This is an image processing pipeline which prepares raw images of 2 cameras for fish tracking in 'TRex'.

STEP 1:


STEP 5: 
The initial background subtraction method failed because some fish remained stationary for extended periods, causing them to be included in the background. 
This issue primarily occurred during the ramping phases (up_i, d_i), where short time windows were used to build a median background. 
However, these short time windows were necessary due to the changing water levels and the associated displacement of background objects caused by refraction.
