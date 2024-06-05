# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:28:27 2021

@author: Naudascher
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# -----  INPUT  -----------------------------
exp = 1

# Set absolute start frame of experiment -> this frame is crucial!!! it should be exactly 20 min before the first upramping event !!!
# chose the frame using fiji or the side cams or the excel sheet

start_second_of_experiment = 124                #94 # These seconds are elapsed in the corweview recording before the 20' acclim time starts... The entire experimental phases are determiend here.

# -------------------------------------------

exp_ID = 'exp_' + str(exp)

fps = 15            # the original recording was at 15 fps

dur_peak_base = 6*60*fps # constant accross all exp
dur_dQ =        1*60*fps   # 1 min ramping duration

if exp == 1:
    in_dir = os.path.join(r'G:\runs_2020\Final',exp_ID,'top_cam')  # Input folder
    start_acclim = start_second_of_experiment * fps # The input folder already conatains this framerate!!
    dur_acclim =   20*60*fps # identical for all experiments (as long as the recording was started sufficiently early!!)

phases = 'p_2'
list_all = os.listdir(in_dir) # all frames

def subtractMedian(start_frame_phase, dur out_dir_phase):
    list_phase = list_all[start_frame_phase : start_frame_phase + dur]     # contains all frames of period
    list_FFT= list_phase[0:100]        # Contains all frames of that period
    
    if not os.path.exists(out_dir_phase):
        os.makedirs(out_dir_phase)
    
    count = 0
    frames = []
    
    # Grab images for calc of FFT
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
    R_zero = 240 #200        #125 # rescaling value for zero values
    R_min = 1           # new min value in rescaled range
    R_max = 255         # 255 new max value in rescaled range
    count = 0
    
    # subtract median fro all frames
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
    
            print('writing frame: ',count)
            count += 1
            print(os.path.join(out_dir_phase, i))
    
            continue
        else:
            continue

for phase in phases: 
        out_dir_phase = os.path.join(out_dir,phase)                 # Output folder fro each phase
   
        if phase == 'p_2':
            start_frame_phase = start_acclim + dur_acclim + dur_dQ + dur_peak_base + dur_dQ + dur_peak_base + dur_dQ + 1
            dur = dur_peak_base
            subtractMedian(start_frame_phase, dur, out_dir_phase)
        
def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

data = np.load('./data.npy')

data.shape

fs = 2.5e6  # sampling frequency, (Hz)
dx = 1      # spatial sampling step along X in (mm)
dy = 1      # spatial sampling step along Y in (mm)

y_max = dy * data.shape[0]  # mm
x_max = dx * data.shape[1]  # mm
t_max = data.shape[2] / fs  # s

y = np.linspace(0, y_max, data.shape[0])   # mm
x = np.linspace(0, x_max, data.shape[1])   # mm

yy, xx = np.meshgrid(y, x, indexing='ij')
time_stamp = 250 # µs
plt.figure()
plt.pcolormesh(xx, yy, data[:,:,time_stamp])
plt.xlabel('x, mm')
plt.ylabel('y, mm')

spectrum_3d = np.fft.fftn(data)                            # Fourrier transform alon Y, X and T axes to obtain ky, kx, f

spectrum_3d_sh = np.fft.fftshift(spectrum_3d, axes=(0,1))  # Apply frequency shift along spatial dimentions so
                                                           # that zero-frequency component appears at the center of the spectrum
                                                           
ky = np.linspace(-np.pi / y_max, np.pi / y_max, data.shape[0])  # wavenumber along Y axis (rad/mm)
kx = np.linspace(-np.pi / x_max, np.pi / x_max, data.shape[1])  # wavenumber along X axis (rad/mm)
f  = np.linspace(0, fs, data.shape[2])                          # frequency (Hz)

Ky, Kx = np.meshgrid(ky, kx, indexing='ij')

freq_to_observe = 40e3     # Hz
f_idx = find_nearest_idx(f, freq_to_observe)
plt.figure()
psd = plt.pcolormesh(Kx, Ky, abs(spectrum_3d_sh[:,:,f_idx])**2)
cbar = plt.colorbar(psd, label='PSD')
plt.xlabel('kx, rad/mm')
plt.ylabel('ky, rad/mm')