#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import quaternion
import pandas as pd
import os
import sys
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import scipy.io as sio
from scipy.signal import find_peaks, butter, filtfilt, iirfilter
from scipy.stats import ttest_ind
#sys.path.append('../../')
#print(sys.path)
print(os.getcwd())
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.signal import find_peaks
from scipy.signal import fir_filter_design as ffd
from scipy.signal import kaiserord, lfilter, firwin, freqz
from math import sqrt, pi


# In[2]:


#insert specfic subject info
subj = '201'
take = 1
loc = 'pre-surgery'
label = '_pre.txt'


# In[3]:


#Change to MC10 path specfic to where box is located on your computer 
mc10_path = r'C:\Users\schne\Box\CMU_MBL\Data\Pitt_Navio_TKA_MC10\Navio'
#need to add r in front of PC path 


# In[4]:


#This is the main function for aligning and syncing data, which calls other fuctions below, 
def load_align_sync_data(path, subj, loc, output_fs):

    # Loads sensor data
    thigh_raw_left = load_sensor_data(path, subj, loc, "left_lateral_thigh")
    shank_raw_left = load_sensor_data(path, subj, loc, "left_lateral_shank")

    thigh_raw_right = load_sensor_data(path, subj, loc, 'right_lateral_thigh')
    shank_raw_right = load_sensor_data(path, subj, loc, 'right_lateral_shank')
    


    print('Data loaded...')

    thigh_si_left = clean_raw_data(thigh_raw_left, 'thigh', 'left')
    shank_si_left = clean_raw_data(shank_raw_left, 'shank', 'left')
    #recfem_si_left = clean_raw_data(recfem_raw_left, 'recfem', 'left')

    thigh_si_right = clean_raw_data(thigh_raw_right, 'thigh', 'right')
    shank_si_right = clean_raw_data(shank_raw_right, 'shank', 'right')
    #recfem_si_right = clean_raw_data(recfem_raw_right, 'recfem', 'right')

    df_si_list_left = [thigh_si_left, shank_si_left]
    df_si_list_right = [thigh_si_right, shank_si_right]

    print('Data cleaned...')

    df_align_list = []
    for df_left in df_si_list_left:
        df_align_list.append(align_data(df_left, 'left'))
    for df_right in df_si_list_right:
        df_align_list.append(align_data(df_right, 'right'))

    print('Data aligned...')

    data_sync = sync_dfs(df_align_list, output_fs)

    print('Data synced...')

    return data_sync


# In[5]:



def load_sensor_data(path, subj, loc, leg):
    '''
    Load data from MC10 sensor csv files to pandas dataframe
    Depending on location, loads accel & gyro or emg
    '''

    #if loc == 'thigh' or loc == 'shank':
        # Loads both accel and gyro file for thigh and shank
    #use file location for PC
    path_accel = path + '\\' +subj+'\\'+loc+'\\'+leg+'\\'+'accel.csv'
    path_gyro = path +'\\'+subj+'\\'+'\\'+loc+'\\'+leg+'\\'+ 'gyro.csv'
    
    #file location for mac 
    #path_accel = path+'/'+subj+'/'+loc+'/'+leg+'/'+'accel.csv'
    #path_gyro = path+'/'+subj+'/'+'/'+loc+'/'+leg+'/'+ 'gyro.csv'
    
    raw_accel = pd.read_csv(path_accel, index_col='Timestamp (microseconds)')
    raw_gyro = pd.read_csv(path_gyro, index_col='Timestamp (microseconds)')

    df_raw = raw_accel.join(raw_gyro, how='outer')

   # elif loc == 'recfem':
        # Loads only emg file for recfem
       # path_emg = os.path.join(path, subj, day, leg+'_'+loc, 'elec.csv')

        #raw_emg = pd.read_csv(path_emg, index_col='Timestamp (microseconds)')

        #df_raw = raw_emg.copy()

    return df_raw


# In[6]:


def clean_raw_data(df_raw, loc, leg):
    '''
    Converts dataframe to SI units
    Filters dataframe with butterworth filter
    Renames columns to include correct units and segment name
    '''
    #ind, peaks =  find_peaks(-df_raw['Accel X (g)'],height=8)
    df_raw = df_raw.interpolate(limit_area='inside')
    #print(df_raw)
    #end = len(df_raw)
    #df_raw = df_raw[ind[0]:end]
    df_si = df_raw.copy()
    df_temp = df_raw.copy()
    df_filt = df_raw.copy()

    new_cols = []

    for col in df_raw.columns:
        if 'Accel' in col:
            # Converts accelerations in [g] to [m/s^2]
            df_si.loc[:, col] = df_raw.loc[:, col] * -9.81
            df_filt.loc[:, col] = butter_low(df_si.loc[:, col].values, order=4, fc=5)

            tmp = col.replace('(g)', ('(m/s^2) '+leg+' '+loc))
            if 'None' in leg:
                tmp = tmp.replace(' '+leg, '')
            new_cols.append(tmp)
        elif 'Gyro' in col:
            # Converts angular velocities in [deg/s] to [rad/s]
            df_si.loc[:, col] = df_raw.loc[:, col] / 180 * np.pi
            #df_temp.loc[:, col] = highpass_iir(df_si.loc[:, col].values, order=1, fc=.25)
            df_filt.loc[:, col] = butter_low(df_si.loc[:, col].values, order=8, fc=10)
            #df_filt.loc[:, col] = fir_low(df_temp.loc[:, col].values, order=8, fc=30)
            tmp = col.replace('(°/s)', ('(rad/s) '+leg+' '+loc))
            if 'None' in leg:
                tmp = tmp.replace(' '+leg, '')
            new_cols.append(tmp)
        elif 'Sample' in col:
            # Keeps activations in [V]
            df_filt.loc[:, col] = butter_low(df_si.loc[:, col].values, order=4, fc=5)

            tmp = col.replace('(V)', ('(V) '+leg+' '+loc))
            if 'None' in leg:
                tmp = tmp.replace(' '+leg, '')
            new_cols.append(tmp)

    df_filt.columns = new_cols

#     #Converts timestamp index to datetime index
#     df_filt['Datetime'] = pd.to_datetime(df_filt.index*1000)
#     df_filt = df_filt.set_index('Datetime')

    # Plot to check the effect of filtering (check)
    #fig, ax_arr = plt.subplots(1, 3)
    #labels = ['X', 'Y', 'Z']
    #labels_filt = ['X - filt', 'Y - filt', 'Z - filt']
    #for i in range(0, 3):
         #idx = i
         #ax_arr[i].plot(df_si.values[:, idx], label=labels[i])
         #ax_arr[i].plot(df_filt.values[:, idx], label=labels_filt[i])
         #ax_arr[i].legend()
    #plt.show()
    return df_filt


# In[7]:


def butter_low(data, order, fc, fs=125):
    '''
    Zero-lag butterworth filter for column data (i.e. padding occurs along axis 0).
    The defaults are set to be reasonable for standard optoelectronic data.
    '''
    # Filter design
    b, a = butter(order, 2*fc/fs, 'low')
    # Make sure the padding is neither overkill nor larger than sequence length permits
    padlen = min(int(0.5*data.shape[0]), 200)
    # Zero-phase filtering with symmetric padding at beginning and end
    filt_data = filtfilt(b, a, data, padlen=padlen, axis=0)
    return filt_data


# In[8]:


def align_data(df, leg):
    '''
    Aligns MC10 data to standard coordinate system
    X - forward; Y - up; Z - right
    '''
    if leg == 'left':
        y_rot_quat = quaternion.from_euler_angles(0, np.pi, 0)
    if leg == 'right':
        y_rot_quat = quaternion.from_euler_angles(0, 0, 0)

    z_rot_quat = quaternion.from_euler_angles(0, 0, np.pi/2)

    rot_quat = z_rot_quat*y_rot_quat
    rot_quat = rot_quat.conj()

    # Segments, sensor types, and coordinates to rotate
    if df.columns[0].find('thigh') != -1:
        seg = 'thigh'
    if df.columns[0].find('shank') != -1:
        seg = 'shank'
    sensors = ['Accel', 'Gyro']
    axes = ['X', 'Y', 'Z']

    for sens in sensors:
        if sens == 'Accel':
            units = '(m/s^2)'
        if sens == 'Gyro':
            units = '(rad/s)'

        col_list = []
        for ax in axes:
            col_list.append(' '.join([sens, ax, units, leg, seg]))
        data = df.loc[:, col_list].values.copy()
        data = quaternion.rotate_vectors(rot_quat, data, axis=1)
        df.loc[:, col_list] = data

    return df


# In[9]:


#this is the sync function that I was having issues with the conversion to timestamp, if we want to, we could 
#un-comment the portion which creates a new index in timestamps, but i dont know that it matters.
#also  sometimes syncing takes longer than the other functions
def sync_dfs(df_list, resamp_freq):
    '''
    Synchronizes dataframes in df_list to start and stop at same index
    Resamples dataframe to desired frequency
    '''

    # Joins all the dataframes in df_list
    col_len = []
    col_start = [0]
    df_len = len(df_list)
    for i in range(df_len):
        col_len.append(df_list[i].shape[1])
        col_start.append((col_start[i]+col_len[i]))
        if i == 0:
            df = df_list[i]
        else:
            df = df.join(df_list[i], how='outer')

    # Interpolate missing timepoints and remove beginning and end NaNs
    df = df.interpolate(limit_area='inside')
    df = df.dropna()

#     # Converts index to datetime if in timestamps
#     if df.index.dtype == 'int64':
#         df['Datetime'] = pd.to_datetime(df.index*1e6)
#         df = df.set_index('Datetime')

#     # Create separate time index at desired frequency
#     freq_in_ms = int(1000/resamp_freq)
#     tmp_idx = pd.date_range(start=df.index[0], end=df.index[-1], freq=(str(freq_in_ms)+'ms'))
#     tmp_idx= pd.Series(np.arange(df.index[0],df.index[-1],freq_in_ms))
#     d = np.zeros((len(tmp_idx)))

#     tmp_df = pd.DataFrame({'temp': d}, index=tmp_idx)
#     tmp_idx.name = 'Datetime'
    
#     # Join inner and outer dataframe to add new indices
#     df = df.join(tmp_df, how='outer')

#     # Drop temporary zeros column
#     df = df.drop(columns=['temp'])

#     # Interpolate data again
#     df = df.interpolate(limit_area='inside')

#     # Extract only the time points of the desired frequency
#     df = df.loc[tmp_idx, :]

#     # Creates dataframe list with synchronized and resampled data
#     df_list_new = []
#     df_add = tmp_df.copy()
#     for i in range(df_len):
#         df_add = tmp_df.copy()

#         for j in range(col_len[i]):
#             df_add = df_add.join(df.loc[:, df.columns[col_start[i]+j]], how='outer')
#             if j == col_len[i]-1:
#                 df_add = df_add.drop(columns=['temp'])

#         df_list_new.append(df_add)
#     time = np.arange(0,len(df_list)/freq_in_ms,1/freq_in_ms)
    return df #_list_new


# In[10]:


#after messing with all the functions, I have been calling the main function here
syncdata = None
syncdata = load_align_sync_data(mc10_path, subj, loc, 125)


# In[11]:


syncdata.index -= syncdata.index[0]
syncdata.index /= 10**6
print(syncdata.index)


# In[21]:


# plot raw gyroscope data

plt.plot(syncdata.index, syncdata.iloc[:,11], color = 'maroon')
plt.xlabel('Time (hours)')
plt.ylabel('Angular velocity (rad/s)')
plt.title('Gyroscope Data in Z-Direction for Left Shank IMU')


# In[ ]:


peaks, _=find_peaks(np.array(syncdata.iloc[:,11]), height = [3,7], distance =100)
plt.subplots()
plt.plot(syncdata.iloc[:,11])
plt.plot(syncdata.iloc[peaks,11],  'x', label='Peak')
plt.legend()


# In[ ]:


def gait_identify(peaks):
    counter = 0
    peaks_temp = []
    start = []
    end = []
    for i in range(len(peaks)-1):
        distance = peaks[i+1] - peaks[i]
        if distance > 800:
            if counter >= 15:
                peaks_temp = np.append(peaks_temp,peaks[(i-counter):i])
                end = np.append(end,peaks[i])
                start = np.append(start, peaks[i-counter])
                counter = 0
            elif counter < 15:
                counter = 0
                continue
        elif distance < 800:
                counter += 1
    return peaks_temp, start, end
peaks_temp, start, end = gait_identify(peaks)
print(start)


# In[ ]:


def between(l1,low,high):
    l2 = []
    for i in l1:
        if(i > low and i < high):
            l2.append(i)
    return l2


# In[ ]:


#plot walking time periods 
m=1
peaks_sub = between(peaks, start[m], end[m])
plt.subplots()
plt.plot(syncdata.iloc[int(start[m]):int(end[m]),11])
plt.plot(syncdata.iloc[peaks_sub,11],  'x', label='Peak')
plt.xlabel('Time (seconds)')
plt.ylabel('Angular Velocity in Z Direction of Left Shank(rad/s)')
plt.title('One Identified Period of Walking (>20 gait cycles)')
plt.legend()


# In[ ]:


plt.subplots()
plt.plot(syncdata.iloc[:,11])
plt.plot(syncdata.iloc[peaks_temp,11],  'x', label='Peak')
plt.legend()


# In[ ]:


def save_gait_cycles(syncdata, mc10_path, subj,loc, start, end):
    length = start.shape[0]
    for i in range(0,length):
        temp = syncdata.iloc[int(start[i]):int(end[i]),: ]
        filename = 'gait_cycle_' + str(i) +'_.csv'
        path = mc10_path+'\\'+subj+'\\'+loc+'\\' + filename  #PC path
        temp.to_csv(path)
        
save_gait_cycles(syncdata, mc10_path, subj, loc, start, end)

