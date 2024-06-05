# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 18:49:20 2021

@author: Naudascher

old version: use step_5_new
"""

import numpy as np
import cv2
import os
import pandas as pd
# from matplotlib import pyplot as plt

# -----  INPUT  -----------------------------
experiments = [6]#,7,8] #[1,5,6,7,8] # rerun

batch_1_wild = True

run_length_soft = (6*6 + 6*3) *60
run_length_rough =(6*6 + 6*1) *60

if batch_1_wild: 
    df = pd.read_excel(r'G:\runs_2020\master_files\batch_1_wild_master.xlsx')
    experiments = ['exp_1','exp_2','exp_3','exp_4','exp_5','exp_6','exp_7','exp_8','exp_9','exp_10']
    backgrounds = ['back_soft_1','back_soft_2','back_steep_1']
# -------------------------------------------


phases = ['b_1']#,'p_1'] # ['acclim','up_1','p_1','d_1','b_1','up_2','p_2','d_2','b_2','up_3','p_3','d_3','b_3',]


for exp in experiments:
    # Set absolute start frame of experiment -> this frame is crucial!!! it should be exactly 20 min before the first upramping event !!!
    # chose the frame using fiji or the side cams or the excel sheet
   # if exp ==1:
    #    start_second_of_experiment = 124                #94 # These seconds are elapsed in the corweview recording before the 20' acclim time starts... The entire experimental phases are determiend here.
    
    if exp == 5:
        start_second_of_experiment = 648
        
    elif exp == 6:
        start_second_of_experiment = 632
        
    elif exp == 7:
        start_second_of_experiment = 828
        
    elif exp == 8:
        start_second_of_experiment = 669
    
    
    exp_ID = 'exp_' + str(exp)
    
    # Output
    out_dir = os.path.join(r'G:\runs_2020\Final',exp_ID,'phases')
    
    # fps -> how many fps did ou use in this analysis? -> 3 fps in this case!
    #fps = 3
    fps = 15            # the original recording was at 15 fps
    
    dur_peak_base = 6*60*fps # constant accross all exp
    
    if exp == 1 or exp == 5 or exp == 6 or exp == 7 or exp == 8: # steep ramp runs
        steep_ramp = True
        in_dir = os.path.join(r'G:\runs_2020\Final',exp_ID,'top_cam')  # Input folder
        start_acclim = start_second_of_experiment * fps # The input folder already conatains this framerate!!
        dur_acclim =   20*60*fps # identical for all experiments (as long as the recording was started sufficiently early!!)
    
    
    if steep_ramp:
        dur_dQ =        1*60*fps   # 1 min ramping duration
    else:
        dur_dQ =        3*60*fps   # 3 min ramping duration
        
    
    
    list_all = os.listdir(in_dir) # all frames
    
    def subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase):
        list_phase = list_all[start_frame_phase : start_frame_phase + dur]     # contains all frames of period
        list_median = list_phase[0::frame_skip_median]        # Contains all frames for median of that period
        
        if not os.path.exists(out_dir_phase):
            os.makedirs(out_dir_phase)
        
        count = 0
        frames = []
        
        # Grab images for calc of Median
        for i in (list_median):
        
            if i.endswith(".tif"):
                img = cv2.imread(os.path.join(in_dir, i))  # Load frame
                img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                frames.append(img_grey)  # Collect all frames
                count += 1
                print('appending frame for Median: ', count)
                continue
            else:
                continue
        
        # Calculate Median Background
        median = np.median(frames, axis=0).astype(dtype=np.uint8)     #cv2.imshow('median Baseflow', median)  #cv2.waitKey(0)
        #maximum = np.max(frames,axis=0).astype(dtype=np.uint8)
        #maximum_grey  = maximum.astype(np.int32) 
        median_grey = median.astype(np.int32)                           # enlarge for Fre
    
        ## Subtract Median; # Fre approach -> Params to rescale the bitdepth after subtraction and reconvert it to type: uint8
        R_zero = 240 #240        #125 # rescaling value for zero values
        R_min = 1           # new min value in rescaled range
        R_max = 255         # 255 new max value in rescaled range
        count = 0
        
        # subtract median for all frames
        for i in (list_phase):
            if i.endswith(".tif"):
                frame = cv2.imread(os.path.join(in_dir, i))
                
                # Fre approach
                frame_grey = np.array(cv2.cvtColor(np.uint8(frame), cv2.COLOR_RGB2GRAY))
                frame_grey = frame_grey.astype(np.int32)  # enlarge frame
                cor_frame = np.array(frame_grey -  median_grey)  # subtract median
                # Reshift pixel values tp positive values
                scale_frame = cor_frame + R_zero
                scale_frame = np.where(scale_frame < R_min, R_min, scale_frame)
                scale_frame = np.where(scale_frame > R_max, R_max, scale_frame)
        
                # RN
                # scale_frame = np.where(frame_grey > median_grey, R_max, scale_frame)
        
                scale_frame = np.uint8(scale_frame)
    
                 # write frame to output folder
                cv2.imwrite(os.path.join(out_dir_phase, i), scale_frame)
        
                print(exp_ID + ' ;writing frame: ',count)
                count += 1
                print(os.path.join(out_dir_phase, i))
        
                continue
            else:
                continue
    
    for phase in phases: 
            out_dir_phase = os.path.join(out_dir,phase)                 # Output folder for each phase
            
            # BASEFLOWS: --------------------------------------------------
            if phase == 'acclim':
                start_frame_phase = start_acclim   # set start frame and phase
                dur = dur_acclim  # all frames you wanna grab    
                frame_skip_median = 100  # These frames will be skipped when calc. the median! (every ith' frame will be used)
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            if phase == 'b_1':
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + 1
                dur = dur_peak_base
                frame_skip_median = 15
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            if phase == 'b_2':
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + 1
                dur = dur_peak_base
                frame_skip_median = 15
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            if phase == 'b_3':
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + 1
                dur = dur_peak_base
                frame_skip_median = 15
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            # PEAKFLOWS: --------------------------------------------------------
            if phase == 'p_1':
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + 1
                dur = dur_peak_base
                frame_skip_median = 15
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            if phase == 'p_2':
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + 1
                dur = dur_peak_base
                frame_skip_median = 15
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
            
            if phase == 'p_3':
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ+ 1
                dur = dur_peak_base
                frame_skip_median = 15
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            # UP-Ramps: -----------------------------------------------------------------
            if phase == 'up_1':
                frame_skip_median = 3
                dur = int(dur_dQ/3) # Here we subdivide the phase into 3 subphases for which we derive their individual background!!!
                # sub_phase_1
                start_frame_phase = start_acclim + dur_acclim + 1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                 # sub_phase_2
                start_frame_phase = start_acclim + dur_acclim + dur + 1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                # sub_phase_3
                start_frame_phase = start_acclim + dur_acclim + 2*dur + 1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
              
            if phase == 'up_2':
                frame_skip_median = 3
                dur = int(dur_dQ/3)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + 1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur + 1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + 2* dur+ 1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            if phase == 'up_3':
                frame_skip_median = 3
                dur = int(dur_dQ/3)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + 1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur +1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + 2*dur + 1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            # Downramps: -----------------------------------------------------------------
            if phase == 'd_1':
                frame_skip_median = 3
                dur = int(dur_dQ/3)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + 1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur +1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + 2*dur +1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            if phase == 'd_2':
                frame_skip_median = 3
                dur = int(dur_dQ/3)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + 1  
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur + 1  
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + 2*dur + 1  
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
               
            if phase == 'd_3':
                frame_skip_median = 3
                dur = int(dur_dQ/3)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base  + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + 1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base  + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur + 1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base  + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + 2*dur +1
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
          
