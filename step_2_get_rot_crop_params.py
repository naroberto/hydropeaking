# -*- coding: utf-8 -*-
"""
STEP_2
pipeline for exp_2020
Created on Tue Aug 24 16:04:36 2021

@author: Naudascher
"""

# adapted from https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/

# Input: stitched .tif
# Output: rotated and cropped image

# GOAL:
# - set upper left corner of the illuminated arena as origin for rotation,
# - select 5 or more points along the horizontal shore-line (downstream direction going to the right)
# - calc. the average rotation angle based on this
# - Rot. image
# - Crop image to size: width =  ; height =
# Note: The center of rotation remains identical for all experiments! (x=1850, y=46) its location will therefore be identical on each image. even across experiments.
# Check opencv version when running this code! -> should be: v 4.5.1.48
import cv2
import numpy as np
import os


# -----  INPUT  -------
# Batch_1_wild: 1,3,5,9    (for one run each day, we create crop rot params)
# Batch_2_wild: 11,14,17,20
# Batch_3_wild:     25,29,34,39,

# Batch_1_hatchery: 41,45,50,55,59,64,67, DONE

# batch_2_hatchery: 70,75,80 

exp = 80

batch_1_wild = False
batch_2_wild = False
batch_3_wild = False

batch_1_hatchery = False

batch_2_hatchery = True
# ---------------------

if batch_1_wild: # on harddisk: 1_Results
    drive ='G:'
    
if batch_2_wild: # on harddisk: 2_Results
    drive = 'L:'
    
if batch_3_wild: 
    if exp >=34:
        drive = 'G:' # on harddisk: 1_Results
    else:
        drive = 'L:' # on harddisk: 2_Results

if batch_1_hatchery:
    drive = 'E:' # on harddisk: 3_Results
    
if batch_2_hatchery:
    drive = 'F:'
        
    
exp_ID = 'exp_' + str(exp)


# Select respective baseflow image for calibration.
#in_path = r'F:\Batch_1_wild\Exp_1_tiff_analysis\stitched\00001.tif'   # low flow image from this run
in_path =             os.path.join(drive,r'runs_2020\Final', exp_ID,r'params\final_stitched_trimmed.tif')

# out path
rot_crop_params_out = os.path.join(drive,r'runs_2020\Final', exp_ID,r'params\rot_crop_params.txt')
rot_crop_tif_out =    os.path.join(drive,r'runs_2020\Final', exp_ID,r'params\final_stitched_rot_crop.tif')

# keep cropping extent - Do never change!
h = 850
w = 2260

# set area to be set to 0 below edge to avoid reflections on the wall
# delta_Y = 20 # Pixel

# GLOBAL VARIABLES
x_cord = []
y_cord = []
rot_angle_all = [] # angle between horizontal and shoreline

# Function to display and store the coordinates of Rot_center and the rot_angle_
# First click -> center for rotation, click at the upper left corner  of the arena
# Clicks after -> upper right edge of the arena, this will be horizontal after...
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        x_cord.append(x) # List with all x-coordinates
        y_cord.append(y) # List with all y-coordinates

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)

        cv2.imshow('set clicks', img)
        # calc rotation angle between side and horizontal
        if len(x_cord) >= 2  :
            deltaY = y - y_cord[0]
            deltaX = x - x_cord[0]
            angle = np.arctan(deltaY / deltaX) * 180 / np.pi
            rot_angle_all.append(angle)



def rotate_keep_orig_dim(image, angle, cX, cY):  # see also: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    # grab the dimensions of the image
    height, width = img.shape[:2]
    # grab the rotation matrix (applying angle to rotate counter clockwise)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # perform the actual rotation and return the image, size of the image wo'nt have changed
    return cv2.warpAffine(image, M, (width,height))


# driver function
if __name__=="__main__":

    # reading the image
    img = cv2.imread(in_path, 1)
    height, width = img.shape[:2]
    cv2.namedWindow('set clicks', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('set clicks', width, height)
    cv2.imshow('set clicks', img)

    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('set clicks', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

    # Calc mean rotation angle
    rot_angle_mean = round(np.mean(rot_angle_all),7)
    print('rot_angle_mean',rot_angle_mean)

img = cv2.imread(in_path, 1)
height, width = img.shape[:2]
cv2.namedWindow('rotated', cv2.WINDOW_NORMAL)
cv2.resizeWindow('rotated', width, height)
#rotated = imutils.rotate(img, rot_angle_mean) # rotates around center of image!!
rotated = rotate_keep_orig_dim(img, rot_angle_mean,x_cord[0],y_cord[0])

cv2.imshow("rotated", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# calc cropping extent relative to Rotation center! These values are for the Topview surveyance cam; derive manually from fiji so that extetn is visible
x_R = x_cord[0] # Rotational center
y_R = y_cord[0]

# this is the translation of the origin
d_y = 15
d_x = 15

# this will be the upper left corner (0,0) after cropping, expressed in coordinates before cropping
y = y_R - d_y # increase extent from y coord of rotational center so that entire height is visible
x = x_R - d_x

# This is the rotational center in coordinates after cropping and rotation
y_R_crop = d_y
x_R_crop = d_x

# Crop the image
rotated_crop = rotated[y:y+h, x:x+w]
height, width = rotated_crop.shape[:2]
cv2.namedWindow('rotated_cropped_highflow', cv2.WINDOW_NORMAL)
cv2.resizeWindow('rotated_cropped_highflow', width, height)
cv2.imshow('rotated_cropped_highflow', rotated_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(rot_crop_tif_out, rotated_crop)

# Set Black stripe
# Set values to black (0,0,0)
# Low-Flow: set the stripe on the lower edge to "0" so we actually get rid of the reflections at the wall
#height, width = rotated_crop.shape[:2]
#rotated_crop[y_R_crop:,:] = [0,0,0]
#rotated_crop_blacked = rotated_crop
#cv2.namedWindow('rotated_crop_blacked', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('rotated_crop_blacked', width, height)
#rotated = imutils.rotate(img, rot_angle_mean) # rotates around center of image!!
#cv2.imshow("rotated_crop_blacked", rotated_crop_blacked)
#cv2.waitKey(0)

# Print to console
print('First point = selected Rotation Point (x,y) : ', x_R, ',', y_R) # + ',' + str(y_cord(1)))
print('All rotation angles: ', rot_angle_all)
print('Mean rotation angle: ', rot_angle_mean)
print('Rotational center/ Reference point in cropped image (x,y) : ', x_R_crop, ',', y_R_crop)
print('Dimensions of cropped image (width,height) : ', w, ',', h)

## WRITE TO .txt

outF = open(rot_crop_params_out, "w")
#for line in [header]: # , basestart]:
outF = open(rot_crop_params_out, "a")
outF.write("rotation center x-coordinate")
outF.write("\n")                # go to next line
outF.write(str(x_R))
outF.write("\n")
outF.write("rotation center y-cordinate")
outF.write("\n")
outF.write(str(y_R))
outF.write("\n")
outF.write("rotation angle")   # write line to output file
outF.write("\n")
outF.write(str(rot_angle_mean))

outF.write("\n")
outF.write("x_R_crop")   # write line to output file
outF.write("\n")
outF.write(str(int(x_R_crop)))

outF.write("\n")
outF.write("y_R_crop")   # write line to output file
outF.write("\n")
outF.write(str(int(y_R_crop)))

outF.write("\n")
outF.write("cropped_width")   # write line to output file
outF.write("\n")
outF.write(str(w))

outF.write("\n")
outF.write("cropped_height")   # write line to output file
outF.write("\n")
outF.write(str(h))

outF.write("\n")
outF.write("x")   # write line to output file
outF.write("\n")
outF.write(str(x))

outF.write("\n")
outF.write("y")   # write line to output file
outF.write("\n")
outF.write(str(y))
outF.close()

#outF.write("\n")
#outF.write("blackened_stripe_height")   # write line to output file
#outF.write("\n")
#outF.write(str(d_y))
#outF.close()                # close file
