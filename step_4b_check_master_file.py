# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:40:39 2022
@author: Naudascher

Goal: Plot hydraulic data. Here we ensured temporal synchronization, between hydraulic data and camera recordings. 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

plt.rc('font', **font)

batch_1_wild = False
batch_2_wild=  False
batch_3_wild = True

batch_1_hatchery = False
batch_2_hatchery = False

saving = True
out_dir = r'G:\varp_plots'

run_length_soft = (6*6 + 6*3) *60
run_length_rough =(6*6 + 6*1) *60

# masterfilepath
df = pd.read_excel(r'G:\runs_2020\master_files\master_file_2020.xlsx')

if batch_1_wild: 
    experiments = [1,2,3,4,5,6,7,8,9,10]
if batch_2_wild: 
    experiments = [11,12,13,14,15,16,17,18,19,20,21,22,23,24]
if batch_3_wild: 
    experiments = [25,26,27,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
if batch_1_hatchery: 
    experiments = [41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69] 
if batch_2_hatchery: 
    experiments = [70,71,72,73,74,75,76,77,78,79,80,81,82,83,84]
                             
for exp in experiments: 
    ID = 'exp_' + str(exp)
    exp_props = df[df['ID']==ID]
    varp_path = exp_props.loc[:, 'varp_path'].apply(str).squeeze()
    treatment = exp_props.loc[:, 'treatment'].apply(int).squeeze()
    start =     exp_props.loc[:, 'varp_acclim_start_second'].apply(int).squeeze()
    acclim_dur_seconds = exp_props.loc[:, 'acclim_dur_min'].apply(int).squeeze() *60
    if treatment == 1: # soft
        run_dur = acclim_dur_seconds + run_length_soft
    elif treatment == 2: # rough
        run_dur = acclim_dur_seconds + run_length_rough
        
    # start figure
    with open(varp_path) as f:
        first_line  = f.readline()
        second_line = f.readline()
        #print(first_line)
        print('column order '+ str(ID) +':  ' + second_line)

    # load data
    data = pd.read_csv(varp_path, sep=None, header = None, skiprows = 2, engine='python')
    #print(str(len(data)))
    #print(data)
    
    #pumpf = data.iloc[start:start+run_dur,1]
    #MID   = data.iloc[start:start+run_dur,2]
    Hoehe = data.iloc[start:start+run_dur,3]
    
    fig_width  = 25
    fig_height = 10
    
    fig = plt.figure(figsize=(fig_width,fig_height))
    plt.title(ID + ' Treatment: ' + str(treatment) + '    check if MID and hoehe are right well labeled',fontsize = 18)
    #plt.plot(pumpf,label = "Pumpfrequency")
    #plt.plot(MID,label = "MID [l/s]")
    plt.plot(Hoehe,label = "Hoehe [cm]")
    plt.axvline(x=start+acclim_dur_seconds,label = "START",color="black")
    #plt.ylim(5,14) # this also inverts the y-achsis
    plt.legend(fancybox=True, framealpha=0.5,loc='upper left',fontsize = 18)
    plt.ylabel('variable')
    plt.show()
    if saving: fig.savefig(os.path.join(out_dir,ID + '.png'), format='png', dpi=300) 

