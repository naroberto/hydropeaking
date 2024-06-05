# -*- coding: utf-8 -*-
"""
STEP_1
pipeline for exp_2020

Created on Mon Aug 23 16:29:02 2021
copied form previous version in pycharm
@author: Naudascher
"""
# Stitch tiff files together: adapted from: https://medium.com/analytics-vidhya/image-stitching-with-opencv-and-python-1ebd9e0a6d78
# Goal: Create transformation matrix M used for stitching
# this is done for each day of experiments, the output file will be used for all the other runs on that same day

import cv2
import numpy as np
import os
#from tempfile import TemporaryFile
import matplotlib.pyplot as plt

# This is unique for each time the camera positions changed, we will use it for the entire experiment. redo it just with one frame in this script
# -----  INPUT  -------
exp = 80


batch_1_wild = False
batch_2_wild = False
batch_3_wild = False

batch_1_hatchery = False
batch_2_hatchery = True

# Batch_1_wild:     1,3,5,9
# Batch_2_wild:     11: on other hard disk,14,17,20
# Batch_3_wild:     25,29,34,39,

# Batch_1_hatchery: 41,45,50,55,59,64,67,
# bacth_2_hatchery: 70,75,80 

# -----------------------

# DO NOT CHANGE THESE!
# RN shift left image by shift_x further to the right!!! Try to keep it low but the stones should somehow transition smoothly.
# I came up with it because the selected fix point for stitching are in the gravel, the code therefore tries ti match the locations in the gravel, this results in mismatching stones
shift_dx = 10 # Trade-off between matching gravel bed vs. large boulders
# cut out central bright line. cut away dx_ pixel from the overlaying right edge of the left image
dx_reduced_edge = 60


exp_ID = 'exp_' + str(exp)
M_filename = 'M_stitch_params.npy'

# output folder
if batch_1_wild:
    out_path = os.path.join(r'G:\runs_2020\Final',exp_ID,'params') # Harddisk name: 1_Results

if batch_2_wild or batch_3_wild:
    if exp >=34: 
        out_path = os.path.join(r'G:\runs_2020\Final',exp_ID,'params') # Harddisk name: 1_Results
    else:
        out_path = os.path.join(r'L:\runs_2020\Final',exp_ID,'params') # Harddisk name : 2_Results
        
if batch_1_hatchery:
    out_path = os.path.join(r'E:\runs_2020\Final',exp_ID,'params') # Harddisk name: 3_Results


if batch_2_hatchery:
    out_path = os.path.join(r'F:\runs_2020\Final',exp_ID,'params') # Harddisk name: 4_Results

# output files
M_out = os.path.join(out_path,M_filename)
orig_matches = os.path.join(out_path,'matches.tif')
pre_dist =     os.path.join(out_path,'pre_dist.tif')
orig_stitched =os.path.join(out_path,'stitched.tif')
orig_img_stitched_crop = os.path.join(out_path,'final_stitched_trimmed.tif')

# create out directory

if not os.path.exists(out_path):
    os.makedirs(out_path)
    
# Batch 1 wild    
"""   
# Batch 1 wild this needs to be done for each day... 
if exp == 1: 
    img_ = cv2.imread(r'H:\Batch_1\14_07_2020\exp_1\Cam1\CoreView_1_Cam1_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'H:\Batch_1\14_07_2020\exp_1\Cam2\CoreView_1_Cam2_02212.tif')  ## LEFT IMAGE
   
elif exp == 3: 
    img_ = cv2.imread(r'H:\Batch_1\15_07_2020\exp_3\Cam1\CoreView_4_Cam1_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'H:\Batch_1\15_07_2020\exp_3\Cam2\CoreView_4_Cam2_02212.tif')  ## LEFT IMAGE
   
# ruff
elif exp == 5: 
    img_ = cv2.imread(r'H:\Batch_1\16_07_2020\exp_5\Cam1\CoreView_2_Cam1_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'H:\Batch_1\16_07_2020\exp_5\Cam2\CoreView_2_Cam2_02212.tif')  ## LEFT IMAGE
  
# ruff
elif exp == 9: 
    img_ = cv2.imread(r'H:\Batch_1\17_07_2020\exp_9\Cam1\CoreView_2_Cam1_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'H:\Batch_1\17_07_2020\exp_9\Cam2\CoreView_2_Cam2_02212.tif')  ## LEFT IMAGE

"""

# Batch_2 wild
"""
if exp == 11: 
    img_ = cv2.imread(r'I:\Batch_2\21_07\exp_11\Cam1\CoreView_6_Cam1_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'I:\Batch_2\21_07\exp_11\Cam2\CoreView_6_Cam2_02212.tif')  ## LEFT IMAGE

elif exp == 14:
    img_ = cv2.imread(r'H:\Batch_2\22_07\exp_14\Cam3\CoreView_9_Cam3_02212.tif') 
    img =  cv2.imread(r'H:\Batch_2\22_07\exp_14\Cam4\CoreView_9_Cam4_02212.tif') 
    
    
elif exp == 17:
    img_ = cv2.imread(r'H:\Batch_2\23_07\exp_17\Cam3\CoreView_10_Cam3_02212.tif') 
    img =  cv2.imread(r'H:\Batch_2\23_07\exp_17\Cam4\CoreView_10_Cam4_02212.tif') 
    
elif exp == 20:
    img_ = cv2.imread(r'H:\Batch_2\24_07\exp_20\Cam3\CoreView_10_Cam3_02212.tif') 
    img =  cv2.imread(r'H:\Batch_2\24_07\exp_20\Cam4\CoreView_10_Cam4_02212.tif') 
"""    
"""  

# Batch_3 wild: 25,29,34,38,
if exp == 25: 
    img_ = cv2.imread(r'I:\Batch_3\28_07\exp_25\Cam3\CoreView_14_Cam3_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'I:\Batch_3\28_07\exp_25\Cam4\CoreView_14_Cam4_02212.tif')  ## LEFT IMAGE

elif exp == 29:
    img_ = cv2.imread(r'I:\Batch_3\29_07\exp_29\Cam3\CoreView_2_Cam3_02212.tif') 
    img =  cv2.imread(r'I:\Batch_3\29_07\exp_29\Cam4\CoreView_2_Cam4_02212.tif') 
    
    
elif exp == 34:
    img_ = cv2.imread(r'I:\Batch_3\30_07\exp_34\Cam1\CoreView_2_Cam1_02212.tif') 
    img =  cv2.imread(r'I:\Batch_3\30_07\exp_34\Cam2\CoreView_2_Cam2_02212.tif') 
    
elif exp == 39:
    img_ = cv2.imread(r'I:\Batch_3\31_07\exp_39\Cam1\CoreView_3_Cam1_02212.tif') 
    img =  cv2.imread(r'I:\Batch_3\31_07\exp_39\Cam2\CoreView_3_Cam2_02212.tif') 
"""
"""
# Batch_1 Hatchery 41,45,50,55,59,64,67

if exp == 41: 
    img_ = cv2.imread(r'J:\Batch_1_hatchery\03_08\exp_41\Cam1\CoreView_2_Cam1_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'J:\Batch_1_hatchery\03_08\exp_41\Cam2\CoreView_2_Cam2_02212.tif')  ## LEFT IMAGE

elif exp == 45:
    img_ = cv2.imread(r'J:\Batch_1_hatchery\04_08\exp_45\Cam1\CoreView_2_Cam1_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'J:\Batch_1_hatchery\04_08\exp_45\Cam2\CoreView_2_Cam2_02212.tif') 

elif exp == 50:
    img_ = cv2.imread(r'J:\Batch_1_hatchery\05_08\exp_50\Cam1\CoreView_2_Cam3_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'J:\Batch_1_hatchery\05_08\exp_50\Cam2\CoreView_2_Cam4_02212.tif') 
    
elif exp == 55:
    img_ = cv2.imread(r'J:\Batch_1_hatchery\06_08\exp_55\Cam3\CoreView_2_Cam3_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'J:\Batch_1_hatchery\06_08\exp_55\Cam4\CoreView_2_Cam4_02212.tif') 
    
elif exp == 59:
    img_ = cv2.imread(r'J:\Batch_1_hatchery\07_08\exp_59\Cam3\CoreView_2_Cam3_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'J:\Batch_1_hatchery\07_08\exp_59\Cam4\CoreView_2_Cam4_02212.tif')   
    
elif exp == 64:
    img_ = cv2.imread(r'K:\Batch_1_hatchery\08_08\exp_64\Cam3\CoreView_2_Cam3_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'K:\Batch_1_hatchery\08_08\exp_64\Cam4\CoreView_2_Cam4_02212.tif') 
    
elif exp == 67:
    img_ = cv2.imread(r'K:\Batch_1_hatchery\10_08\exp_67\Cam3\CoreView_2_Cam3_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'K:\Batch_1_hatchery\10_08\exp_67\Cam4\CoreView_2_Cam4_02212.tif') 
    #K:\Batch_1_hatchery\08_08\exp_64\Cam3

"""


# Batch_2 Hatchery

if exp == 70:
    img_ = cv2.imread(r'K:\Batch_2_hatchery\11_08\exp_70\Cam3\CoreView_2_Cam3_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'K:\Batch_2_hatchery\11_08\exp_70\Cam4\CoreView_2_Cam4_02212.tif')  ## LEFT IMAGE

elif exp == 75:
    img_ = cv2.imread(r'K:\Batch_2_hatchery\12_08\exp_75\Cam3\CoreView_2_Cam3_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'K:\Batch_2_hatchery\12_08\exp_75\Cam4\CoreView_2_Cam4_02212.tif') 

elif exp == 80:
    img_ = cv2.imread(r'K:\Batch_2_hatchery\13_08\exp_80\Cam1\CoreView_2_Cam1_02212.tif') ### RIGHT IMAGE rewrite this...  try: use 02212 for setting params
    img =  cv2.imread(r'K:\Batch_2_hatchery\13_08\exp_80\Cam2\CoreView_2_Cam2_02212.tif')  
#%%

# The same needs to be done for the background !!!


# Background
#img_ = cv2.imread(r'G:\Batch_1\16_07_2020\CoreView_1\Cam1\CoreView_1_Cam1_0255.tif')
# Background
# img = cv2.imread(r'G:\Batch_1\16_07_2020\CoreView_1\Cam2\CoreView_1_Cam2_0255.tif')


# Greyscale
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create() # no clue what this function actually does

# find key points
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# Show keypoints
#cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))
#cv2.waitKey(100)

# Use this method to find similar gradients in both imgs
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
search_params = dict(checks = 100000)
match = cv2.FlannBasedMatcher(index_params, search_params)

#match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2)
good = []

# remove all matches that are not more or less on a horizontal line & are to close or too far a prt to make actual sense
def filterMatches(kp1, kp2, matches, diff_y_max, diff_x_min, diff_x_max):

# Adpated by RN: (from: https://stackoverflow.com/questions/33499254/filtering-sift-points-by-y-coordinate-with-opencv-python)
# Removes the matches that correspond to a pair of keypoints (kp1, kp2)
# which y-coordinate difference is lower than imgHeight * thresFactor.
#
# Args:
#     kp1 (array of cv2.KeyPoint): Key Points.
#
#     kp2 (array of cv2.KeyPoint): Key Points.
#
#     matches (array of cv2.DMATCH): Matches between kp1 and kp2.
#
#     diff_y_max: max vertical shift of matching points (unit = pixels)
#
#     diff_x_min: min distance in x of matching points  (unit = pixels)
#
#     diff_x_max: max distance in x of matching points  (unit = pixels)
#
# Returns:
#     array of cv2.DMATCH: filtered matches.
    filteredMatches = [None]*len(matches)
    counter = 0
    #threshold = imgHeight * thresFactor
    for i in range(len(kp1)):
        srcPoint = kp1[ matches[i][0].queryIdx ].pt
        dstPoint = kp2[ matches[i][0].trainIdx ].pt
        diff_y = abs(srcPoint[1] - dstPoint[1])
        diff_x = abs(srcPoint[0] - dstPoint[0]) # calc abs distance
        
        if (diff_y < diff_y_max) & (diff_x < diff_x_max) & (diff_x > diff_x_min) & (srcPoint[1]< half_height ):
            filteredMatches[counter] = matches[i]
            counter += 1

    return filteredMatches[:counter]

half_height = 1024*0.9 # don't use matches on the very deep side of the flume, -> hight distortion there
filtered_matches = filterMatches(kp1, kp2, matches, 50,1000,2000 )

for m,n in filtered_matches:
    if m.distance < 0.8 * n.distance: # this is the goodness of fit criteria regarding the selected fit points points 0.8 kindo worked
        good.append(m)

draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       flags=2)

img3 = cv2.drawMatches(img_,kp1,img,kp2,good,None,**draw_params)
cv2.imshow('used_matches_for_stitching', img3)
# wait for a key to be pressed to exit
cv2.waitKey(0)
# close the window
cv2.destroyAllWindows()
cv2.imwrite(orig_matches, img3)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # this approach chnages the perspective!!
    M_new = cv2.estimateAffine2D(src_pts,dst_pts,False) # this is an affine transformation, it does not touch scalingor perspective to in order to match the points!
    M = np.vstack([M_new[0], [0,0,1]])
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #dst = cv2.perspectiveTransform(pts, M)
    #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    #cv2.imshow("original_image_overlapping.jpg", img2)
else:
    print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))

print('img.shape[1].  ',img.shape[1])


np.save(M_out, M) # save perpective Matrix transform
dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0]))
print('dst.shape[1]: ',dst.shape[1])
#cv2.imwrite("pre_dst.jpg", dst)

# Move the right image a bit further to the right manually, the light refraction makes the edge appear puzzling
# RN shift image by shift_x further to the right!!!
dst[0:dst.shape[0], shift_dx:dst.shape[1]] = dst [0:dst.shape[0], 0:dst.shape[1] - shift_dx]
cv2.imwrite(pre_dist, dst)

print('dst.shape[1]: ',dst.shape[1])
#cv2.imwrite(pre_dist,img)

# To avoid a white edge in the center of the combine image, we don't use the entire left image!

dst[0:img.shape[0],0:img.shape[1]-dx_reduced_edge] = img[0:img.shape[0],0:img.shape[1]-dx_reduced_edge] # insert the left part by overwriting the left part of dst
cv2.imshow('orig_stitched', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(orig_stitched, dst)

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

trimmed = trim(dst)
cv2.imshow('orig_img_stitched_crop',trimmed)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(orig_img_stitched_crop,trimmed)
print('final dimensions: ',trimmed.shape)

