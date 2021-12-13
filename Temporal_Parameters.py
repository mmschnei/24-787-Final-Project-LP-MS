#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import statistics

from statsmodels.tsa.seasonal import seasonal_decompose


# In[109]:


def find_csv_filenames( path_to_dir, suffix ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if suffix in filename ]
    


# In[317]:


subj = '211'
loc = 'pre-surgery'
#Change to MC10 path specfic to where box is located on your computer 
mc10_path = '/Users/laurenparola/Box/CMU_MBL/Data/Pitt_Navio_TKA_MC10/Navio'

sub_path =  mc10_path+'/'+subj+'/'+loc+'/'
string = 'gait'
ans = find_csv_filenames(sub_path,string)
fulllist = []
j=0
counter = 1
for i in ans:
    alldata = pd.read_csv(sub_path+i)
    rshankz = alldata['Gyro Z (rad/s) right shank']
     #plt.plot(range(len(rshankz)),rshankz)
    end = int(rshankz.index[-1])
    lshankz = alldata['Gyro Z (rad/s) left shank']
    label = pd.DataFrame({'Subject Number':[subj+str(counter)]})
    params_df, mean_std_df, gait_cycle_mat_left, gait_cycle_mat_right = sensor_gait_param_analysis(lshankz, rshankz, np.arange(0,end,1/200), output_fs=200)
    result = pd.concat([label, mean_std_df], axis=1, join="outer") 
    if j == 0:
         fulllist = result
    elif j > 0:
         fulllist = pd.concat([fulllist, result], axis=0,join="outer")    
    j+=1
    counter += 1


# In[318]:


print(fulllist)


# In[ ]:





# In[319]:



fullvec = []
ticker = 0
counter = 1
for i in ans:
    alldata = pd.read_csv(sub_path+i)
    headers = alldata.columns.values
    temp = []
    ticker2 = 0
    for j in headers[1:len(headers)]:
        vectemp = alldata[j]
        maxval = max(vectemp)
        minval = min(vectemp)
        meanval = statistics.mean(vectemp)
        stdval = statistics.stdev(vectemp)
        rangeval = maxval-minval
        tempvec = pd.DataFrame({'Max '+j:[maxval],'Minimum '+j:[minval],'Mean '+j:[meanval],'Standard Deviation '+j:[stdval]})
        if ticker2 == 0:
            activity = pd.DataFrame({'Subject Number':[subj+str(counter)]})
            temp = pd.concat([activity, tempvec], axis=1)
        elif ticker2 > 0:
            temp = pd.concat([temp, tempvec], axis=1)
        ticker2 += 1
    if ticker == 0:
        fullvec = temp
    elif ticker > 0:
        fullvec = fullvec.append(temp)
    ticker +=1
    counter += 1


# In[320]:



print(fullvec)


# In[321]:


bothvec = pd.concat([fullvec,fulllist], axis=1)
bothvec = bothvec.loc[:,~bothvec.columns.duplicated()]
print(np.shape(bothvec))


# In[322]:


print(bothvec)
print(np.shape(bothvec))


# In[323]:


bothvec.to_csv(sub_path+'/'+subj+loc+'_features.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[251]:




def mean_std(data):
    return np.mean(data), np.std(data)



def get_asymmetry(swt_left, swt_right):
    '''
    Calculates limb asymmetry
    '''
    # Calculating using averages of swt (paper method)

    swtl_mean, swtl_std = mean_std(swt_left/1e3)
    swtr_mean, swtr_std = mean_std(swt_right/1e3)

    if swtl_mean > swtr_mean:
        lswt_mean = swtl_mean
        sswt_mean = swtr_mean
    else:
        lswt_mean = swtr_mean
        sswt_mean = swtl_mean

    asym_paper = abs(np.log(sswt_mean/lswt_mean))

    # Calculate using individual swing times

    lswt = []
    sswt = []
    asym = []

    for i in range(len(swt_left)):
        if swt_left[i] > swt_right[i]:
            lswt.append(swt_left[i])
            sswt.append(swt_right[i])
        else:
            lswt.append(swt_right[i])
            sswt.append(swt_left[i])

        asym.append(abs(np.log(sswt[i]/lswt[i])))

    asym_mean, asym_std = mean_std(asym)

    #print('Asymmetry with means vs all')
    #print(asym_paper)
    #print(asym_mean)
    #print(asym_std)
    #print()


# In[252]:


def sensor_gait_param_analysis(lshank_z_df, rshank_z_df, analysis_time, output_fs=125):
    '''
    Takes left and right shank Z-gyroscope data and outputs gait parameters
    '''
    lshank_z = lshank_z_df.values
    rshank_z = rshank_z_df.values
    lheel_contact_time, ltoe_contact_time = extract_contact_points(lshank_z, analysis_time)
    rheel_contact_time, rtoe_contact_time = extract_contact_points(rshank_z, analysis_time)

    # plt.show()

    # IC(L) -> TC(R) -> IC(R) -> TC(L) -> IC(L) + 1
    gait_cycle_mat_left = extract_gait_cycle(
        lheel_contact_time, ltoe_contact_time, rheel_contact_time, rtoe_contact_time)
    # IC(R) -> TC(L) -> IC(L) -> TC(R) -> IC(R) + 1
    gait_cycle_mat_right = extract_gait_cycle(
        rheel_contact_time, rtoe_contact_time, lheel_contact_time, ltoe_contact_time)

#    test1 = gait_cycle_mat_left[:, 0]
#    test2 = gait_cycle_mat_left[:, 1]
#    test3 = gait_cycle_mat_left[:, 2]
#    test4 = gait_cycle_mat_left[:, 3]
#    plt.subplots()
#    plt.plot(lshank_z_df, label='left')
#    plt.plot(test1, lshank_z_df[test1], 'x')
#    plt.plot(test4, lshank_z_df[test4], 'x')
#    plt.plot(rshank_z_df, label='right')
#    plt.plot(test2, rshank_z_df[test2], 'x')
#    plt.plot(test3, rshank_z_df[test3], 'x')
#    plt.title('Locations of Left Gait cycle mat sensors')

#     test1 = gait_cycle_mat_right[:, 0]
#     test2 = gait_cycle_mat_right[:, 1]
#     test3 = gait_cycle_mat_right[:, 2]
#     test4 = gait_cycle_mat_right[:, 3]
#     plt.subplots()
#     plt.plot(lshank_z_df, label='left')
#     plt.plot(test2, lshank_z_df[test2], 'x')
#     plt.plot(test3, lshank_z_df[test3], 'x')
#     plt.plot(rshank_z_df, label='right')
#     plt.plot(test1, rshank_z_df[test1], 'x')
#     plt.plot(test4, rshank_z_df[test4], 'x')
#     plt.title('Locations of right Gait cycle mat sensors')
    # plt.show()

    # Extracts gait cycle times (GCT), swing times (swt), swing percents (swp), double support (ds), and limp
    gct_left, swt_left, swt_r_left, swp_left, ds_left, limp_left = extract_gait_params(
        gait_cycle_mat_left)
    gct_right, swt_right, swt_l_right, swp_right, ds_right, limp_right = extract_gait_params(
        gait_cycle_mat_right)

    left_pdf, left_msdf = get_gait_param_means(
        'left', 'sensor', gait_cycle_mat_left[:, 0], gct_left,
        swt_left, swp_left, ds_left, limp_left)
    right_pdf, right_msdf = get_gait_param_means(
        'right', 'sensor', gait_cycle_mat_right[:, 0], gct_right,
        swt_right, swp_right, ds_right, limp_right)
    # get_asymmetry(swt_left, swt_right)

    params_df = pd.concat([left_pdf, right_pdf], axis=1)
    mean_std_df = pd.concat([left_msdf, right_msdf], axis=1)

    return params_df, mean_std_df, gait_cycle_mat_left, gait_cycle_mat_right


# In[253]:


def extract_gait_cycle(heel_contact, toe_contact, adj_heel_contact, adj_toe_contact):
    '''
    Takes contact times for both this leg and adjacent leg
    Generates gait cycles in the form of:
        initial contact of the target leg [IC(T)]
        to terminal contact of the adjacent leg [TC(A)]
        to IC(A)
        to TC(T)
        to IC(T)+1
    For one leg
    '''

    gait_cycle_mat = []

    # Starts cycle at target heel contact
    for ic_r in heel_contact[:, 0]:
        gait_cycle_temp = [ic_r]

        if ic_r != heel_contact[-1]:
            # Checks for next adjacent toe contact
            atc=np.argmax(adj_toe_contact>ic_r)
            att = adj_toe_contact[atc,0]
            gait_cycle_temp.append(att)
            #Checks for next adjacent heel contact
            ahc=np.argmax(adj_heel_contact>ic_r)
            aht = adj_heel_contact[ahc,0]
            gait_cycle_temp.append(aht)
             #Checks for next target toe contact
            tc=np.argmax(toe_contact>ic_r)
            tct = toe_contact[tc,0]
            gait_cycle_temp.append(tct)
            # Checks for next target heel contact
            nhc=np.argmax(heel_contact>ic_r)
            nht = heel_contact[nhc,0]
            gait_cycle_temp.append(nht)
        else:
            break

        if gait_cycle_temp[-1] - gait_cycle_temp[0] >= 2000:
            continue
        else:
            for i in range(len(gait_cycle_temp)-1):
                if gait_cycle_temp[i] > gait_cycle_temp[i+1]:
                    continue
                elif i+1 == len(gait_cycle_temp)-1:
                    gait_cycle_mat.append(np.array(gait_cycle_temp))

    gait_cycle_mat = np.array(gait_cycle_mat)
    #print(gait_cycle_mat)
   # maxval = np.argmax(gait_cycle_mat,axis=1) == 4:
           #     temp = np.append(temp, gait_cycle_mat[i,:],axis=0)
            #elif np.argmax(gait_cycle_mat[i,:],axis=0) != 4:
               # continue
                
    #fig22, ax_arr22 = plt.subplots()
    #ax_arr22.plot(gait_cycle_mat[:, 0], gait_cycle_mat[:, 0], '^')
    #ax_arr22.plot(gait_cycle_mat[:, 1], gait_cycle_mat[:, 1], '>')
    #ax_arr22.plot(gait_cycle_mat[:, 2], gait_cycle_mat[:, 2], 'v')
    #ax_arr22.plot(gait_cycle_mat[:, 3], gait_cycle_mat[:, 3], '<')
    #plt.title('Gait cycle points sequentially')
    #gait_cycle_mat = temp
    return gait_cycle_mat


# In[254]:


def extract_gait_params(gait_cycle_mat):
    '''
    Outputs gait cycle time, swing time, and swing percent from gait cycles
    '''
    # Determine the gait cycle time (GCT)
    length = len(gait_cycle_mat)
    gct = gait_cycle_mat[:, 4] - gait_cycle_mat[:, 0]
    #adjgct = gait_cycle_mat[2:length, 3]-gait_cycle_mat[1:length-1, 3]
    # Calculate swing times and percentages
    swing_times = gait_cycle_mat[:, 4] - gait_cycle_mat[:, 3]
    adj_swing_times = gait_cycle_mat[:, 2] - gait_cycle_mat[:, 1]
    swing_percents = swing_times / gct * 100

    # Calculate double support (DS) with initial (IDS) and terminal (TDS)
    ids = (gait_cycle_mat[:, 1] - gait_cycle_mat[:, 0]) / gct * 100
    tds = (gait_cycle_mat[:, 3] - gait_cycle_mat[:, 2]) / gct * 100

    ds = ids + tds

    # Calculate limp
    limp = np.abs(ids-tds)

    return gct, swing_times, adj_swing_times, swing_percents, ds, limp


# In[255]:


def extract_contact_points(shank_z, analysis_time, output_fs=125):
    '''
    Outputs the initial (heel) and terminal (toe) contact points for sensor data
    '''
    min_peak_dist = output_fs*0.4
    # min_height = 0.174

    # Finds the top peaks and the minimum peaks
    swing_peak, _ = find_peaks(shank_z, height=2.1, distance=min_peak_dist)
    min_peak, _ = find_peaks(-1*shank_z,height=.35)

    heel_contact = []  # Initial (after peak)
    toe_contact = []  # Terminal (before peak)
    #print(min_peak)
    for sp in swing_peak:
        for mp_forward in min_peak:
            if mp_forward > sp:
                heel_contact.append(mp_forward)
                break
        for mp_backward in reversed(min_peak):
            if mp_backward < sp:
                toe_contact.append(mp_backward)
                break

    #plt.subplots()
    #plt.plot(range(len(shank_z)),shank_z)
    #plt.plot(swing_peak, shank_z[swing_peak], 'x',label='Swing Peak')
    #plt.plot(min_peak, shank_z[min_peak], 'x', label='Min Peak')
    #plt.plot(heel_contact, shank_z[heel_contact], 'o',label='Heel Contact')
    #plt.plot(toe_contact, shank_z[toe_contact], 'o',label='Toe off')
    #plt.title('Heel/Toe contact in O, Swing and Min in X')
    #plt.legend()
    heel_contact = np.array(heel_contact).reshape((-1, 1))
    toe_contact = np.array(toe_contact).reshape((-1, 1))
    heel_contact_time = analysis_time[heel_contact]
    toe_contact_time = analysis_time[toe_contact]

    return heel_contact_time, toe_contact_time


# In[256]:


def get_gait_param_means(leg, method, start_time, gct, swt, swp, ds, limp):
    '''
    Gets mean and standard deviations of gait params for specific leg
    '''

    length = len(gct)
    leg_list = [leg for i in range(length)]
    method_list = [method for i in range(length)]
    param_df = pd.DataFrame(
        {'Leg': leg_list, 'Method': method_list, 'Cycle Start': start_time, 'GCT': gct/1e3, 'Swing Time': swt, 'Swing Percent': swp, 'Double Support': ds, 'Limp': limp})

    gct_mean, gct_std = mean_std(gct)
    swt_mean, swt_std = mean_std(swt)
    swp_mean, swp_std = mean_std(swp)
    ds_mean, ds_std = mean_std(ds)
    limp_mean, limp_std = mean_std(limp)
    

    mean_std_list = [gct_mean, gct_std, swp_mean, swp_std, ds_mean, ds_std, limp_mean, limp_std]
    mean_std_labels = ['Gait cycle time avg. [s] '+leg,'Gait cycle time std. [s] '+leg,
                       'Swing time avg. [% GCT] '+leg, 'Swing time std. [% GCT] '+leg,
                       'Double support avg. [% GCT] '+leg,' Double support std. [% GCT] '+leg,
                       'Limp avg. [% GCT] '+leg, 'Limp std. [% GCT] '+leg]
    mean_std_df = pd.DataFrame([mean_std_list], columns=mean_std_labels)

#     print()
#     print(leg + ' leg: ')
#     print(' Gait cycle time [s]:        ' + str(gct_mean) + ' +/- ' + str(gct_std))
#     print(' Avg. swing time [% GCT]:   ' + str(swp_mean) + ' +/- ' + str(swp_std))
#     print(' Double Support [% GCT]:    ' + str(ds_mean) + ' +/- ' + str(ds_std))
#     print(' Limp [% GCT]:              ' + str(limp_mean) + ' +/- ' + str(limp_std))
    return param_df, mean_std_df


# In[ ]:





# In[ ]:





# In[ ]:




