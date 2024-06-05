# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 19:00:33 2021

@author: Naudascher
Goal open each varp file and copy path to master excel file
- mark if data is weird
- 
"""
import pandas as pd
import matplotlib.pyplot as plt
# from datetime import datetime

exp = 17
#fpath = 'H:\Flume_Wild_Fish_June_2020\Varp_Data\24_Jun_MID_PumpFr_Hoehe_P1'
# Batch_1
if exp == 1:
    file_path = r'H:\Varp\Batch_1\14_07\Treat1_14_Jul_Pumpfr_MID_Hoehe_P1.txt'
    
elif exp == 2 :
    file_path = r'H:\Varp\Batch_1\14_07\Treat2_14_Jul_Pumpfr_MID_Hoehe_P1.txt'
    
elif exp == 3 or exp ==4:
    file_path = r'H:\Varp\Batch_1\15_07\EXP4_15_Jul_Pumpfr_MID_Hoehe_P2.txt'
    
elif exp == 5 or exp == 6 or exp == 7 or exp == 8:
    file_path = r'H:\Varp\Batch_1\16_07\16_Jul_all_Pumpfr_MID_Hoehe_P2.txt'

elif exp == 9 or exp == 10:
    file_path = r'H:\Varp\Batch_1\17_07\17_Jul_all_Pumpfr_MID_Hoehe_P2.txt'
    
elif exp == 17: # corrected
    file_path = r'H:\Varp\Batch_2\23_Jul_all_Pumpfr_MID_Hoehe_P2.txt'   
    

with open(file_path) as f:
    first_line = f.readline()
    second_line=f.readline()
    #print(first_line)
    print(second_line)

data = pd.read_csv(file_path, sep=None, header = None, skiprows = 2, engine='python')
print(str(len(data)))
print(data)

#pumpf = data.iloc[:,1]
#MID   = data.iloc[:,2]
Hoehe = data.iloc[3340:3350,3]


#t = pd.dataframe[0:len(data)]
#print(t)
#plt.plot(pumpf,label = "Pumpfrequency")
#plt.plot(MID,label = "MID")
plt.plot(Hoehe,label = "Hoehe")
#plt.axvline(x=1775)
plt.legend()
plt.ylabel('variable')
plt.title('exp_' +str(exp) + '    check if MID and hoehe are right well labeled')
plt.show()
#

# Get start of hohe:
print('start of Hoehe: ', data.iloc[1775,0])

