# -*- coding: utf-8 -*-
"""
--- STEP 5 alternative ---
Created on Mon Apr  4 14:08:26 2022
@author: Naudascher

Description: 
Here we can obtian another video version of up_i and d_i (up and down-ramping phases) that can be used for tracking in case the previous version did not work well

-> use these videos only if you see that the fish is covering itself during the ramp... synchronization with the background recording is not always ideal!!!

# GOALS
# same as step_5_new.py but for up and down phase, here we want to subtract the background that we recorded because fish otheriese coveres itsel if not moving...


# 2. Output:
# - .mp4 videos    (and .tiff frame sequence if needed)

"""
import pandas as pd
import numpy as np
import cv2
import os

# -----  INPUT  -----------------------------
batch_1_wild = True
batch_2_wild = False
batch_3_wild = False

df = pd.read_excel(r'G:\runs_2020\master_files\master_file_2020.xlsx')            # input
out_videos_folder = r'F:\runs_2020\all_output_vids\up_down_synched_background'    # output videos all in one folder!!!

# SELECT EXPERIMENTS: (WE keep out 17, 18, 23, 27, 28, 40)
experiments =  [67,72] # 

# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,  19,20,21,  22,24, 25, 26,     29,30,31,32,33,34,35,36,37,38,39,   41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,84]
# [70,71,72,73,74,75,76,77,78,79,80,81,82,83,84] #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,  19,20,21,  22,24, 25, 26,     29,30,31,32,33,34,35,36,37,38,39,   41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69]
# To do: run it for after checdking time synch
# [70,71,72,73,74,75,76,77,78,79,80,81,82,83,84]

phases = ['up_1','up_2','up_3','d_1','d_2','d_3']
save_video  = True
print('---------------- run for: ' + str(phases) + ' -----------------')

save_raw_frames = False # just to check id wanted 
save_frames = False # not needed for TRex

# -------------------------------------------------------CONSTANTS------------------------------------

n_phases_ramp_median = 5    #  used 10 should be dividable by 3 , subphases for median backgroudn subtractions in ramps
skip_ramp =      1          # or 3    
                            # for long stationary flows we need less frames for the median, use mor if n_phases_ramp_medianis slarger              
                            # These frames will be skipped when calc. the median! (every ith' frame will be used)
fps = 15                    # the original recording was at 15 fps
dur_peak_base = 6*60*fps    # [frame ]constant accross all exp
width =  2260
height = 850                      

# -------------------------------------------------------------------------------------
print('Sub-phases for ramp background subtraction: --> ', n_phases_ramp_median)
print('Frames for Median: rough  ramp: -->             ', 1*60*15/n_phases_ramp_median/skip_ramp)
print('Frames for Median: soft  ramp: -->              ', 3*60*15/n_phases_ramp_median/skip_ramp)

codec = 'MP4V'  # video props

def subtractMedian_up_down(back_start_frame_phase, start_frame_phase, dur, frame_skip_median, out_dir_phase): 
    list_phase = list_all[start_frame_phase : start_frame_phase + dur]     # contains all frames of period
    
    list_median = back_list_all[back_start_frame_phase : back_start_frame_phase + dur : frame_skip_median]        # Contains all frames for median of that period
    print('Appending for median: ' + str(len(list_median)))
    if not os.path.exists(out_dir_phase): os.makedirs(out_dir_phase)
        
    if save_raw_frames:
        if not os.path.exists(out_dir_phase +'_raw'):
            os.makedirs(out_dir_phase +'_raw')
        
    count = 0
    frames = []
    
    # Grab images for calc of Median
    for i in (list_median):
    
        if i.endswith(".tif"):
            img = cv2.imread(os.path.join(back_in_dir, i))  # Load frame
            img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            frames.append(img_grey)  # Collect all frames
            count += 1
            #print(count)
            
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
            
            # if save_raw_frames == True:
            if save_raw_frames == True or phase == 'b_2': # this is only to check if the phase is synchronzided well
                cv2.imwrite(os.path.join(out_dir_phase +'_raw', i), frame)
                #print(exp + '---> writing raw_frame to--->',out_dir_phase +'_raw')
                #print('progress:',count)
            
            if save_frames == True: 
                cv2.imwrite(os.path.join(out_dir_phase, i), scale_frame)
                #print(exp + ' ;writing frame: ',count)
                #print(os.path.join(out_dir_phase, i))
            # write frame to video
            if save_video == True: 
                out_vid.write(scale_frame)#cv2.cvtColor(scale_frame))#, cv2.COLOR_BGR2GRAY))
                #print(exp + ', ' + phase + '  ---> writing video to --->',out_dir_video)
                #print('progress:',count)
            count += 1
            #print(count)
            
    
            continue
        else:
            out_vid.release() # release video
            print('COMPLETE:    ' + exp + ' ' + phase)
            continue

for _exp in experiments: 
    exp = 'exp_' + str(_exp)
    
    # load related properties from excel
    exp_props = df[df['ID']==exp] # this is one row of the excel file
    frames_path_root = exp_props.loc[:, 'frames_path'].apply(str).squeeze()
    treatment = exp_props.loc[:, 'treatment'].apply(int).squeeze()
    start_acclim =     exp_props.loc[:, 'start_second_acclim_cores'].apply(int).squeeze() * fps
    dur_acclim = exp_props.loc[:, 'acclim_dur_min'].apply(int).squeeze() * 60 * fps
    
    # Folders  
    in_dir =  os.path.join(frames_path_root,exp,'top_cam')  # Input Core frames
    out_dir = os.path.join(frames_path_root,exp,'phases')   # Output frames
   
    
    # Final Videos
    out_dir_video = out_videos_folder  #-> 
    # out_dir_video = os.path.join(frames_path_root,exp,'videos') 
    #if not os.path.exists(os.path.join(out_dir_video)): 
    #   os.makedirs(os.path.join(out_dir_video))
    
    # Ramp
    if treatment == 1:   dur_dQ = 3*60*fps   # soft:  3 min ramping duration
    else:                dur_dQ = 1*60*fps   # rough: 1 min ramping duration
    
    # frames
    list_all = os.listdir(in_dir) # all frames
    
    # Background properties     (use seperate background according to ramp and as defined in master_file_2020.xls)
    back = exp_props['background'].apply(str).squeeze()     # background ID to be used for that run
    back_props = df[df['ID']==back]    # properties of background to be used
    back_frames_path = back_props.loc[:, 'frames_path'].apply(str).squeeze() 
    back_start_up = back_props.loc[:, 'start_second_background'].apply(int).squeeze() * fps # the background was on upramping scenario
    
    back_in_dir =  os.path.join(back_frames_path,back,'top_cam')  # path to background images !!!
    print('get frames from: ' + str(back_in_dir))
    # background frames
    back_list_all = os.listdir(back_in_dir) # all Background frames
  
    
    for phase in phases: 
            print('process:        ' + exp +' ' + phase)
            out_dir_phase = os.path.join(out_dir,phase)                 # Output folder for each phase
            print('writing frames to --> ',out_dir_video,exp + '_' +phase +'.mp4')
            
            
            # UP-Ramps: -----------------------------------------------------------------
            if phase == 'up_1':
                frame_skip_median = skip_ramp
                dur = int(dur_dQ/n_phases_ramp_median) # Here we subdivide the phase into n subphases for which we derive their individual background!!!
                # sub_phase_1
                out_vid = cv2.VideoWriter(os.path.join(out_dir_video,exp + '_' +phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) 
                for sub_phase in np.arange(n_phases_ramp_median):
                    start_frame_phase      = start_acclim + dur_acclim + sub_phase*dur + 1 # sub_phase has to be 0 for first...
                    back_start_frame_phase = back_start_up + sub_phase*dur + 1
                    print('process phase: '+ phase + ', sub_phase:' + str(sub_phase))
                    subtractMedian_up_down(back_start_frame_phase, start_frame_phase, dur, frame_skip_median, out_dir_phase)
              
               # adapt the below accordingly!!! 
            if phase == 'up_2':
                frame_skip_median = skip_ramp
                dur = int(dur_dQ/n_phases_ramp_median) # Here we subdivide the phase into 3 subphases for which we derive their individual background!!!
                # sub_phase_1
                out_vid = cv2.VideoWriter(os.path.join(out_dir_video,exp + '_' +phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) 
                for sub_phase in np.arange(n_phases_ramp_median):
                    start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base +sub_phase*dur + 1
                    back_start_frame_phase = back_start_up + sub_phase*dur + 1
                    print('process phase: '+ phase + ', sub_phase:' + str(sub_phase))
                    subtractMedian_up_down(back_start_frame_phase, start_frame_phase, dur, frame_skip_median, out_dir_phase)
                    
                
            if phase == 'up_3':
                frame_skip_median = skip_ramp
                dur = int(dur_dQ/n_phases_ramp_median) # Here we subdivide the phase into 3 subphases for which we derive their individual background!!!
                # sub_phase_1
                out_vid = cv2.VideoWriter(os.path.join(out_dir_video,exp + '_' +phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) 
                for sub_phase in np.arange(n_phases_ramp_median):
                    start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + sub_phase*dur +1
                    back_start_frame_phase = back_start_up + sub_phase*dur + 1
                    print('process phase: '+ phase + ', sub_phase:' + str(sub_phase))
                    subtractMedian_up_down(back_start_frame_phase, start_frame_phase, dur, frame_skip_median, out_dir_phase)
                
            # Downramps: -----------------------------------------------------------------
            if phase == 'd_1':
                frame_skip_median = skip_ramp
                dur = int(dur_dQ/n_phases_ramp_median) # Here we subdivide the phase into 3 subphases for which we derive their individual background!!!
                # sub_phase_1
                out_vid = cv2.VideoWriter(os.path.join(out_dir_video,exp + '_' +phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) 
                for sub_phase in np.arange(n_phases_ramp_median):
                    start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + sub_phase*dur+ 1
                    back_start_frame_phase = back_start_up + (max(np.arange(n_phases_ramp_median)) - sub_phase)*dur + 1      
                    print('process phase: '+ phase + ', sub_phase:' + str(sub_phase))                    # the background was always an up-ramp, therefore we start with the end of the background !!!
                    subtractMedian_up_down(back_start_frame_phase, start_frame_phase, dur, frame_skip_median, out_dir_phase)
                    

            if phase == 'd_2':
                frame_skip_median = skip_ramp
                dur = int(dur_dQ/n_phases_ramp_median) # Here we subdivide the phase into 3 subphases for which we derive their individual background!!!
                # sub_phase_1
                out_vid = cv2.VideoWriter(os.path.join(out_dir_video,exp + '_' +phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) 
                for sub_phase in np.arange(n_phases_ramp_median):
                    start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + sub_phase*dur+ 1  
                    back_start_frame_phase = back_start_up + (max(np.arange(n_phases_ramp_median)) - sub_phase)*dur + 1  
                    print('process phase: '+ phase + ', sub_phase:' + str(sub_phase))                       # the background was always an up-ramp, therefore we start with the end of the background !!!
                    subtractMedian_up_down(back_start_frame_phase, start_frame_phase, dur, frame_skip_median, out_dir_phase)
                           
               
            if phase == 'd_3':
                frame_skip_median = skip_ramp
                dur = int(dur_dQ/n_phases_ramp_median) # Here we subdivide the phase into 3 subphases for which we derive their individual background!!!
                # sub_phase_1
                out_vid = cv2.VideoWriter(os.path.join(out_dir_video,exp + '_' +phase +'.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (width,height),isColor=False) 
                for sub_phase in np.arange(n_phases_ramp_median):
                    start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base  + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + sub_phase*dur+ 1
                    back_start_frame_phase = back_start_up + (max(np.arange(n_phases_ramp_median)) - sub_phase)*dur + 1    
                    print('process phase: '+ phase + ', sub_phase:' + str(sub_phase))                      # the background was always an up-ramp, therefore we start with the end of the background !!!
                    subtractMedian_up_down(back_start_frame_phase, start_frame_phase, dur, frame_skip_median, out_dir_phase)
           
                
out_vid.release() # release very last video
print('-------------------- DONE : script: step_5_new: ' + str(experiments) + str(phases) + ' ----------------- ')

