# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:47:23 2021

@author: Naudascher
"""

# RN: 30/01/2020 adapted 24.08.2021

# Input: raw .tif image streams from two cams that were recording instantaneously.
# Goal: Pipeline converting .tif images of one Experiment
# Run after step_1 and step_2
# stitch and crop based on prederived params (from step_1 and step_2)

# Output: stitched, rotated, cropped .tif stack

import cv2
import numpy as np
import os


# -----  INPUT  -------


batch_1_wild = False
batch_2_wild = False
batch_3_wild = False
batch_1_hatchery = True
batch_2_hatchery = False

# run this for one day only and use parallel threats if needed!

"""
date = '21_07'
experiments = ['back_steep_2','exp_11','exp_12','exp_13'] # DONE
""" 
"""
date = '22_07'
experiments = ['back_soft_3','exp_14','exp_15','exp_16']
""" 
"""
date = '23_07'
experiments = ['back_steep_3','exp_17','exp_18','exp_19']
"""
"""
date = '24_07'
experiments = ['back_soft_4','exp_20','exp_21','exp_22']
"""
"""
date = '25_07'
experiments = ['back_steep_4','exp_23','exp_24']
"""

# -----------------------  batch_3_wild --------------
# input params from: 25,29,34,39,
"""
date = '28_07'
experiments = ['back_steep_5','exp_25','exp_26','exp_27','exp_28']
input_params_exp_ID = 'exp_25'


date = '29_07'
experiments = ['exp_33'] # 'back_soft_5','exp_29','exp_30','exp_31','exp_32',
input_params_exp_ID = 'exp_29'


date = '30_07'
experiments = ['back_steep_6','exp_34','exp_35','exp_36','exp_37']
input_params_exp_ID = 'exp_34'


# RUN DATE 31_07 next!!!

date = '31_07'
experiments = ['back_soft_6','exp_38','exp_39','exp_40']
input_params_exp_ID = 'exp_39'
"""


# --------------------- batch_1_hatchery ---------------------
"""
date = '03_08'
experiments = ['back_steep_7','exp_41','exp_42','exp_43','exp_44',]
input_params_exp_ID = 'exp_41'

"""
date = '04_08'
experiments = ['exp_49']#['back_soft_7','exp_45','exp_46','exp_47','exp_48','exp_49']
input_params_exp_ID = 'exp_45'
"""

date = '05_08'
experiments = ['back_steep_8','exp_50','exp_51','exp_52','exp_53','exp_54']
input_params_exp_ID = 'exp_50'


date = '06_08'
experiments = ['back_soft_8','exp_55','exp_56','exp_57','exp_58']
input_params_exp_ID = 'exp_55'


date = '07_08'
experiments = ['back_steep_9','exp_59','exp_60','exp_61','exp_62','exp_63']
input_params_exp_ID = 'exp_59'



date = '08_08'
experiments = ['back_soft_9','exp_64','exp_65','exp_66']
input_params_exp_ID = 'exp_64'


date = '10_08'
experiments = ['back_steep_10','exp_67','exp_68','exp_69']
input_params_exp_ID = 'exp_67'

"""

# ------------------- Batch 2 hatchery --------------------------------------
"""
date = '11_08'
experiments = ['back_steep_11','exp_70','exp_71','exp_72','exp_73','exp_74']
input_params_exp_ID = 'exp_70' # use those for all experiments that day


date = '12_08'
experiments = ['back_soft_10','exp_75','exp_76','exp_77','exp_78','exp_79']
input_params_exp_ID = 'exp_75'


date = '13_08'
experiments = ['back_soft_11','exp_80','exp_81','exp_82','exp_83','exp_84']
input_params_exp_ID = 'exp_80'
"""


# ---------------------------------------------------------------------------------

if batch_1_wild: # on harddisk: 1_Results
    drive ='G:'
    
if batch_2_wild: # on harddisk: 2_Results
    drive = 'L:'
    
if batch_3_wild: 
    if date == '30_07' or date == '31_07':
        drive = 'G:' # on harddisk: 1_Results
    else:
        drive = 'L:' # on harddisk: 2_Results

if batch_1_hatchery: # this is where the params fiels are!
    drive = 'E:' # on harddisk: 3_Results
    
if batch_2_hatchery: # this is where the params fiels are!
    drive = 'F:' # on harddisk: 3_Results
        

for exp in experiments:
    
    exp_ID = exp
    print(' ------------------------  processing: ', exp_ID, '---------------------------------')
    # Output
    out_dir = os.path.join(drive,r'runs_2020\Final',exp_ID,'top_cam')
    
    # Input params
    #M_matrix_path = os.path.join(drive,r'runs_2020\Final',exp_ID,'params\M_stitch_params.npy') # This output was generated in the script "step1_stitch_tiff"
    #params_file = os.path.join(drive,r'runs_2020\Final', exp_ID,r'params\rot_crop_params.txt') # Rotation and final cropping
    
    M_matrix_path = os.path.join(drive,r'runs_2020\Final',input_params_exp_ID,'params\M_stitch_params.npy') # This output was generated in the script "step1_stitch_tiff"
    params_file = os.path.join(drive,r'runs_2020\Final', input_params_exp_ID,r'params\rot_crop_params.txt')
    
    
    # Input images
    if batch_1_wild:
        in_right_dir = os.path.join(r'H:\Batch_1',date,exp_ID,'Cam1') # Right (downstream cam)
        in_left_dir =  os.path.join(r'H:\Batch_1',date,exp_ID,'Cam2') # Left (upstream cam)
        
    if batch_2_wild & (date =='21_07'):
        in_right_dir = os.path.join(r'I:\Batch_2',date,exp_ID,'Cam1') # Right (downstream cam)
        in_left_dir =  os.path.join(r'I:\Batch_2',date,exp_ID,'Cam2') # Left (upstream cam)
    elif batch_2_wild:
        in_right_dir = os.path.join(r'H:\Batch_2',date,exp_ID,'Cam3') # Right (downstream cam)
        in_left_dir =  os.path.join(r'H:\Batch_2',date,exp_ID,'Cam4') # Left (upstream cam)
        
    if batch_3_wild:
        if (date =='28_07') or (date =='29_07'):
            in_right_dir = os.path.join(r'I:\Batch_3',date,exp_ID,'Cam3') # Right (downstream cam)
            in_left_dir =  os.path.join(r'I:\Batch_3',date,exp_ID,'Cam4') # Left (upstream cam)
        else:
            in_right_dir = os.path.join(r'I:\Batch_3',date,exp_ID,'Cam1') # Right (downstream cam)
            in_left_dir =  os.path.join(r'I:\Batch_3',date,exp_ID,'Cam2') # Left (upstream cam)
    
        
    if batch_1_hatchery:
        if (date =='03_08') or (date =='04_08') or('05_08'):
            in_right_dir = os.path.join(r'J:\Batch_1_hatchery',date,exp_ID,'Cam1') # Right (downstream cam)
            in_left_dir =  os.path.join(r'J:\Batch_1_hatchery',date,exp_ID,'Cam2') # Left (upstream cam)
        if (date =='06_08') or (date =='07_08'):
            in_right_dir = os.path.join(r'J:\Batch_1_hatchery',date,exp_ID,'Cam3') # Right (downstream cam)
            in_left_dir =  os.path.join(r'J:\Batch_1_hatchery',date,exp_ID,'Cam4') # Left (upstream cam)
        elif (date =='08_08') or (date =='10_08'):
            in_right_dir = os.path.join(r'K:\Batch_1_hatchery',date,exp_ID,'Cam3') # Right (downstream cam)
            in_left_dir =  os.path.join(r'K:\Batch_1_hatchery',date,exp_ID,'Cam4') # Left (upstream cam)
    
    if batch_2_hatchery:
        in_right_dir = os.path.join(r'K:\Batch_2_hatchery',date,exp_ID,'Cam1') # Right (downstream cam)
        in_left_dir =  os.path.join(r'K:\Batch_2_hatchery',date,exp_ID,'Cam2') # Left (upstream cam)
        
    #K:\Batch_2_hatchery\13_08
    # create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('Write stitched images to: ', out_dir)
    
    # Set framerate 
    ith_frame = 1 # Coreview was recorded at 15 fps, we will use 3 fps in a first step! -> use every 5'th frame !
    
    ## Stitching params
    # Load the Transformation matrix
    M = np.load(M_matrix_path) # load transformation matrix!
    
    # DO NOT CHANGE THESE!
    shift_dx = 10        # RN shift image by shift_x further to the right!!!
    dx_reduced_edge = 60 # cut out central bright line. -> cut away dx_ pixel from the overlaying right edge of the left image
    
    ## Read parameters for rotation and cropping
    with open(params_file) as f:
        contents = f.readlines()
        
    
    x_R = int(contents[1])         # rotation center in original frame, upper left corner is (0,0)
    y_R = int(contents[3])
    rot_angle = float(contents[5])    # angle it will be rotated counter clockwise around this point
    y_R_crop = int(contents[9])
    width = int(contents[11])         # width of cropped image
    height = int(contents[13])        # height of cropped image
    
    # this will be the upper left corner (0,0) after cropping
    x = int(contents[15])
    y = int(contents[17])   # increase extent from y coord of rotational center so that width is visible
    
    ## Image folders
    
    # Create list's with path to images
    list_right = os.listdir(in_right_dir)
    list_right = list_right[0::ith_frame] # use only ith image
    
    list_left = os.listdir(in_left_dir)
    list_left = list_left[0::ith_frame] # use only ith image
    
    print('Total frames to process: ', len(list_left))
    print('Duration of Run in minutes: ', len(list_left)*ith_frame/15/60)
    
    ## Functions
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
    
    # rotation function
    def rotate_keep_orig_dim(img, angle, cX, cY):  # see also: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        # grab the dimensions of the image
        h, w = img.shape[:2]
        # grab the rotation matrix (applying angle to rotate counter clockwise)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        # perform the actual rotation and return the image, size of the image wo'nt have changed
        return cv2.warpAffine(img, M, (w,h))
    
    
    
    # Don't change
    count = 0
    splitat = 16 # here we will split string to get output frame ID
    
    # Stitch all images together
    for r in (list_right):
    
        if r.endswith(".tif"):
    
            # Load frames
            img_ = cv2.imread(os.path.join(in_right_dir, r))               # right img
            img  = cv2.imread(os.path.join(in_left_dir, list_left[count])) # left img
    
            # Convert to grayscale
            img_ = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #print(r)
            #print(list_left[count])
    
            # Transform according to M
            dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0]))
    
            # RN: shift right image by shift_x further to the right!!!
            dst[0:dst.shape[0], shift_dx:dst.shape[1]] = dst [0:dst.shape[0], 0:dst.shape[1] - shift_dx]
    
            # RN: insert the left img on the left
            dst[0:img.shape[0],0:img.shape[1]-dx_reduced_edge] = img[0:img.shape[0],0:img.shape[1]-dx_reduced_edge] # insert the left part by overwriting the left part of dst
            #print('dst.shape[1]: ',dst.shape[1])
    
            # Trim the edges according to dim
            dst = trim(dst)
            # print('dst.shape[1]: ',dst.shape[0])
    
            # Rotate
            rotated = rotate_keep_orig_dim(dst, rot_angle, x_R, y_R)
            #print('rotated dim:',rotated.shape[:2])
    
            # Crop
            rotated_cropped = rotated[y:y+height, x:x+width]
            #print('rotated_cropped dim:',rotated_cropped.shape[:2])
    
            frame_ID = r[splitat:]
            #print(frame_ID)
    
            # write frame to outpu folder
            #cv2.imwrite(out_dir + '\ ' + frame_ID, rotated_cropped)
            cv2.imwrite((os.path.join(out_dir, frame_ID)), rotated_cropped)
            #print((os.path.join(out_dir, frame_ID)))
    
            #print(exp_ID + ' Frame: ', count)
            count += 1
            continue
        else:
            continue
print('-------------------- DONE: step_3:  ----------------- ')
print('Date:        ' + date )
print('Experiments: ' + str(experiments) )