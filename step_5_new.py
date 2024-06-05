# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 18:49:20 2021
@author: Naudascher

GOALS:
- subtract median background for base and peakflows
- subdivide pre-processed images into .mp4 videas of our experimental phases: acclim, up_1, p_1, d_1, b_1, up_2, p_2, d_2, b_1, up_3, p_3, d_3, b_3  
- check if the timing of phases is ok, run it for b_2 and all experiments

Output:
- .mp4 video of experimental phase, for Tracking.
- .tif frame sequence in subsequent folder structure 

"""
import pandas as pd
import numpy as np
import cv2
import os

# -----  INPUT  -----------------------------
check_time_synch = False    # set to False after checking time synch -

# -> open b_2_raw frames in fiji -> actually open the video
# if needed: adapt column: start_second_acclim_cores in ... _master.xlsx file
# ---------------------------------------------

# input master file
df = pd.read_excel(r'E:\master_file_2020.xlsx')  # check that the path to frames is correct in the excel!

# output
out_dir_video =    r'F:\runs_2020\all_output_vids\all_phases' # output videos all in one folder!!!
out_dir_b2_raw =   r'F:\runs_2020\all_output_vids\b2_raw'     # output videos all in one folder!!!

# ALL EXPERIMENTS: don't process due to a variety of errors:      17, 18, 23, 27, 28, 40
# total experiments: 78 !!!
# rerun it because timing was still bad for those:
experiments = [14,15,16,19,20,21,22,24, 25, 26] #[63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84] # [ 29,30,31,32,33,34,35,36,37,38,39, 41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62  , #[55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84] 
              
    # all (groups and individuals):
    #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,  19,20,21,  22,24, 25, 26,  29,30,31,32,33,34,35,36,37,38,39, 41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84]
    
if check_time_synch:
    phases = ['b_2'] # check if b_2 is nicely synchronized
    save_raw_video = True # check raw frames... and see where waves start/end
    save_video  = False
    save_frames = False # not needed for TRex
    save_raw_frames = False
    print('---------------- check_time_synch -----------------')
    print('----write video(s) to:  ',out_dir_b2_raw)
    
else:
    phases = ['d_3','up_3']#['up_2','p_2','d_2','b_2','up_3','p_3','d_3','b_3']  # already done: ['acclim','d_1','up_1','p_1','b_1']#     #,'up_2','p_2','d_2','b_2','up_3','p_3','d_3','b_3'] -> run this later!! for now we focus on fisrt part only!
    save_raw_frames = False # just to check
    save_raw_video = False
    save_video  = True
    save_frames = False # not needed for TRex
    print('---------------- run for all phases -----------------')
    print('----write videos to:  ',out_dir_video)

# -----------------CONSTANTS----------------------

fps = 15                    # the original recording was at 15 fps
dur_peak_base = 6*60*fps    # [frame ]constant accross all exp
n_phases_ramp_median = 5    # should be dividable by 3 , subphases for median backgroudn subtractions in ramps
codec = 'MP4V'              # codec of output video 

# These frames will be skipped when calc. the median! (every ith' frame will be used), increases speed             
skip_acclim =    5         # for long stationary discharge we need less frames for the median             
skip_ramp =      2                                    
skip_base_peak = 3

# ---------------------------------------------
# Show some props: 
print('Sub-phases for ramp background subtraction: --> ', n_phases_ramp_median)
print('Frames for Median: rough  ramp: -->             ', 1*60*15/n_phases_ramp_median/skip_ramp)
print('Frames for Median: soft  ramp: -->              ', 3*60*15/n_phases_ramp_median/skip_ramp)

for _exp in experiments: 
    
    exp = 'exp_' + str(_exp)
    # load bacth related props from excel
    exp_props = df[df['ID']==exp]
    frames_path_root = exp_props.loc[:, 'frames_path'].apply(str).squeeze()
    treatment = exp_props.loc[:, 'treatment'].apply(int).squeeze()
    start_acclim =     exp_props.loc[:, 'start_second_acclim_cores'].apply(int).squeeze() * fps
    dur_acclim = exp_props.loc[:, 'acclim_dur_min'].apply(int).squeeze() * 60 * fps

    # Folders  
    in_dir =  os.path.join(frames_path_root,exp,'top_cam')  # Input Core frames
    out_dir = os.path.join(frames_path_root,exp,'phases')   # Output frames
    
    # for videos
    #out_dir_video = os.path.join(frames_path_root,exp,'videos')
    
    #if not os.path.exists(os.path.join(out_dir_video)):
     #           os.makedirs(os.path.join(out_dir_video))

    if treatment == 1:                  
        dur_dQ =        3*60*fps   # soft:  3 min ramping duration
    else: 
        dur_dQ =        1*60*fps   # rough: 1 min ramping duration
    
    list_all = os.listdir(in_dir) # all frames
        
    def subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase):
        
        out_vid_raw = cv2.VideoWriter(os.path.join(out_dir_b2_raw,exp + '_' + phase +'_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False)
        list_phase = list_all[start_frame_phase : start_frame_phase + dur]     # contains all frames of period
        list_median = list_phase[0::frame_skip_median]        # Contains all frames for median of that period
        
        if not os.path.exists(out_vid_path):
            os.makedirs(out_vid_path)
            
        count = 0
        frames = []
        
        # Grab images for calc of Median
        for i in (list_median):
            if i.endswith(".tif"):
                img = cv2.imread(os.path.join(in_dir, i))  # Load frame
                img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                frames.append(img_grey)  # Collect all frames
                count += 1
                continue
            else:
                continue
                
        #print('frames for Median: ', count)
        # Calculate Median Background
        median = np.median(frames, axis=0).astype(dtype=np.uint8)     #cv2.imshow('median Baseflow', median)  #cv2.waitKey(0)
        #maximum = np.max(frames,axis=0).astype(dtype=np.uint8)
        #maximum_grey  = maximum.astype(np.int32) 
        median_grey = median.astype(np.int32)                           # enlarge for Fre
    
        ## Subtract Median; # Fre approach -> Params to rescale the bitdepth after subtraction and reconvert it to type: uint8
        R_zero = 255 #240        #125 # rescaling value for zero values
        R_min = 1           # new min value in rescaled range
        R_max = 255         # 255 new max value in rescaled range
        count = 0
        
        # subtract median for all frames
        for i in (list_phase):
            
            if i.endswith(".tif"):
                frame = cv2.imread(os.path.join(in_dir, i))
                
                # Rescale pixel values 
                frame_grey = np.array(cv2.cvtColor(np.uint8(frame), cv2.COLOR_RGB2GRAY))
                frame_grey_ = frame_grey.astype(np.int32)  # enlarge frame
                cor_frame = np.array(frame_grey_ -  median_grey)  # subtract median
                
                # Reshift pixel values tp positive values
                scale_frame = cor_frame + R_zero
                scale_frame = np.where(scale_frame < R_min, R_min, scale_frame)
                scale_frame = np.where(scale_frame > R_max, R_max, scale_frame)
                scale_frame = np.uint8(scale_frame)

                if save_raw_video == True:
                    out_vid_raw.write(frame_grey)
                    
                if save_frames == True: 
                    cv2.imwrite(os.path.join(out_dir_phase, i), scale_frame)
                    #print(exp + ' ;writing frame: ',count)
                    #print(os.path.join(out_dir_phase, i))
                
                # write frame to video
                if save_video == True: 
                    out_vid.write(scale_frame)#cv2.cvtColor(scale_frame))#, cv2.COLOR_BGR2GRAY))

                count += 1
                #print(count)
                continue
            else:
                out_vid.release() # release video
                out_vid_raw.release()
                print('COMPLETE:    ' + exp + ' ' + phase)
                continue
    
    for phase in phases: 
            print('process:        ' + exp +' ' + phase)
            out_dir_phase = os.path.join(out_dir,phase)                 # Output folder for each phase
            print('writing frames to --> ',out_dir_video,exp + '_' +phase +'.mp4')
            width =  2260
            height = 850
            
            # BASEFLOWS: --------------------------------------------------
            if phase == 'acclim':
                start_frame_phase = start_acclim   # set start frame and phase
                dur = dur_acclim  # all frames you wanna grab    
                frame_skip_median = skip_acclim  # These frames will be skipped when calc. the median! (every ith' frame will be used)
                out_vid_path = os.path.join(out_dir_video,phase)
                out_vid =     cv2.VideoWriter(os.path.join(out_vid_path,exp + '_' + phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            if phase == 'b_1':
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + 1
                dur = dur_peak_base
                frame_skip_median = skip_base_peak
                out_vid_path = os.path.join(out_dir_video,phase)
                out_vid =     cv2.VideoWriter(os.path.join(out_vid_path,exp + '_' + phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT
                # initialize VIDEO OBJECT
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            if phase == 'b_2':
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + 1
                dur = dur_peak_base
                frame_skip_median = skip_base_peak
                out_vid_path = os.path.join(out_dir_video,phase)
                out_vid =     cv2.VideoWriter(os.path.join(out_vid_path,exp + '_' + phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            if phase == 'b_3':
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + 1
                dur = dur_peak_base
                frame_skip_median = skip_base_peak
                out_vid_path = os.path.join(out_dir_video,phase,exp + '_' + phase +'.mp4')
                out_vid =     cv2.VideoWriter(out_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            # PEAKFLOWS: --------------------------------------------------------
            if phase == 'p_1':
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + 1
                dur = dur_peak_base
                frame_skip_median = skip_base_peak
                out_vid_path = os.path.join(out_dir_video,phase)
                out_vid =     cv2.VideoWriter(os.path.join(out_vid_path,exp + '_' + phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            if phase == 'p_2':
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + 1
                dur = dur_peak_base
                frame_skip_median = skip_base_peak
                out_vid_path = os.path.join(out_dir_video,phase,exp + '_' + phase +'.mp4')
                out_vid =     cv2.VideoWriter(out_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT 
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
            
            if phase == 'p_3':
                start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ+ 1
                dur = dur_peak_base
                frame_skip_median = skip_base_peak
                out_vid_path = os.path.join(out_dir_video,phase)
                out_vid =     cv2.VideoWriter(os.path.join(out_vid_path,exp + '_' + phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT
                subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            # UP-Ramps: -----------------------------------------------------------------
            if phase == 'up_1':
                frame_skip_median = skip_ramp
                dur = int(dur_dQ/n_phases_ramp_median) # Here we subdivide the phase into 3 subphases for which we derive their individual background!!!
                # sub_phase_1
                out_vid_path = os.path.join(out_dir_video,phase)
                out_vid =     cv2.VideoWriter(os.path.join(out_vid_path,exp + '_' + phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT
                for sub_phase in np.arange(n_phases_ramp_median):
                    start_frame_phase = start_acclim + dur_acclim + sub_phase*dur + 1 # sub_phase has to be 0 for first...
                    subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
              
               # adapt the below accordingly!!! 
            if phase == 'up_2':
                frame_skip_median = skip_ramp
                dur = int(dur_dQ/n_phases_ramp_median) # Here we subdivide the phase into 3 subphases for which we derive their individual background!!!
                # sub_phase_1
                out_vid_path = os.path.join(out_dir_video,phase)
                out_vid =     cv2.VideoWriter(os.path.join(out_vid_path,exp + '_' + phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT
                for sub_phase in np.arange(n_phases_ramp_median):
                    start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base +sub_phase*dur + 1
                    subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            if phase == 'up_3':
                frame_skip_median = skip_ramp
                dur = int(dur_dQ/n_phases_ramp_median) # Here we subdivide the phase into 3 subphases for which we derive their individual background!!!
                # sub_phase_1
                out_vid_path = os.path.join(out_dir_video,phase)
                out_vid =     cv2.VideoWriter(os.path.join(out_vid_path,exp + '_' + phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT
                for sub_phase in np.arange(n_phases_ramp_median):
                    start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + sub_phase*dur +1
                    subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            # Downramps: -----------------------------------------------------------------
            if phase == 'd_1':
                frame_skip_median = skip_ramp
                dur = int(dur_dQ/n_phases_ramp_median) # Here we subdivide the phase into 3 subphases for which we derive their individual background!!!
                # sub_phase_1
                out_vid_path = os.path.join(out_dir_video,phase)
                out_vid =     cv2.VideoWriter(os.path.join(out_vid_path,exp + '_' + phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT
                for sub_phase in np.arange(n_phases_ramp_median):
                    start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + sub_phase*dur+ 1
                    subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            if phase == 'd_2':
                frame_skip_median = skip_ramp
                dur = int(dur_dQ/n_phases_ramp_median) # Here we subdivide the phase into 3 subphases for which we derive their individual background!!!
                # sub_phase_1
                out_vid_path = os.path.join(out_dir_video,phase)
                out_vid =     cv2.VideoWriter(os.path.join(out_vid_path,exp + '_' + phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT
                for sub_phase in np.arange(n_phases_ramp_median):
                    start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + sub_phase*dur+ 1  
                    subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
                           
            if phase == 'd_3':
                frame_skip_median = skip_ramp
                dur = int(dur_dQ/n_phases_ramp_median) # Here we subdivide the phase into 3 subphases for which we derive their individual background!!!
                # sub_phase_1
                out_vid_path = os.path.join(out_dir_video,phase)
                out_vid =     cv2.VideoWriter(os.path.join(out_vid_path,exp + '_' + phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) # initialize VIDEO OBJECT
                for sub_phase in np.arange(n_phases_ramp_median):
                    start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base  + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + sub_phase*dur+ 1
                    subtractMedian(start_frame_phase, dur, frame_skip_median, out_dir_phase)
   
out_vid.release()      # release very last video
print('-------------------- DONE : script: step_5_new: ' + str(experiments) + str(phases) + ' ----------------- ')
