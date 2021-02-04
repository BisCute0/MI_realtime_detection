# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:41:21 2020

@author: Cindy Lin

上傳denoised完後的資料至MongoDB
"""

import pymongo
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as sig
import os
import time

data_len_sec = 5
freq = 256
# start = 1605078650

if __name__ == '__main__':
    start = int(time.time())
    print("Current Time:", start)
    SKH_3lead_clean_folder = '.\SKH_Diff_3lead_clean'
    patient_ID = '09597615'
    
    myclient = pymongo.MongoClient("mongodb://192.168.25.22:27017/")
    mydb = myclient['ecg']

def ecg_3lead_insert(mydb, patient_ID, denoised_3lead, start_time):
    '''
    一次丟5秒資料進來，上傳到Mongo
    20201130: 拿到的start_time是5秒資料的結尾，所以上傳的時候要往回5秒
    '''
    
    Diff_1 = denoised_3lead['diff1']
    Diff_2 = denoised_3lead['diff2']
    Diff_3 = denoised_3lead['diff3']
    
    mycol = mydb["ecg3_denoised"]
    start_time = start_time - (data_len_sec - 1) # 往前4秒，讓3轉12拿到的時間是對的
    for i in range(0, data_len_sec):
        Patient_CodeID = patient_ID
        Diff1_list, Diff2_list, Diff3_list = [], [], []              
        Diff1_list = Diff_1[0, freq * i: freq * (i + 1)].tolist()
        Diff2_list = Diff_2[0, freq * i: freq * (i + 1)].tolist()
        Diff3_list = Diff_3[0, freq * i: freq * (i + 1)].tolist()
        
        mydict = {"Patient_CodeID": Patient_CodeID,'Ecg_time': start_time, \
                  'Diff_1': Diff1_list, 'Diff_2': Diff2_list, 'Diff_3': Diff3_list}
        mycol.insert_one(mydict)
        start_time = start_time + 1
        
    # 上傳完denoised資料後，要更新user table
    # lasttime_3denoised統一傳這5秒資料的第一秒的timestamp
    update_time = start_time
    mycol = mydb['user']
    myquery = { 'userId': Patient_CodeID }
    update_data = {'$set': {'lasttime_3denoised': update_time} }
    mycol.update_one(myquery, update_data)
    
    
def mi_result_insert(mydb, patient_ID, diagnosis, start_time):
    '''
    一次丟5秒資料進來，上傳到Mongo
    20201130: 拿到的start_time是5秒資料的結尾，所以上傳的時候要往回5秒
    '''
    result = diagnosis['predict_result']   # 1:positive  0: negative
    mi_type = diagnosis['mi_type']  # {AMI, IMI, None}
    
    if result:
        mycol = mydb["ecg_mi_diagnosis"]
        mydict = {"Patient_CodeID": patient_ID,'Ecg_time': start_time,\
                  "predict_result":result, "mi_type": mi_type}
        mycol.insert_one(mydict)

    # 更新user table
    # is_detected_mi 回傳這5秒資料的第一秒的timestamp
    update_time = start_time
    mycol = mydb['user']
    myquery = { 'userId': patient_ID }
    print('userId:%s' % patient_ID)
    update_data = {'$set': {'is_detected_mi': result,'lasttime_mi_detect':update_time} }
    print('is_detected_mi:%d \t lasttime_mi_detect:%d'% (result,update_time))
    mycol.update_one(myquery, update_data)
    print("Upload user MI")


    
def collect_mongo_data(mycol, start_time, end_time, patient_ID):
    '''
    從MongoDB上抓下ECG raw data
    '''
    myquery = {'Patient_CodeID': str(patient_ID),'Ecg_time': {'$gte': start_time,'$lt': end_time} }
    mydoc = mycol.find(myquery).sort('Ecg_time', pymongo.ASCENDING) # sort by timestamp
    diff1 = np.array([])
    diff2 = np.array([])
    diff3 = np.array([])

    ecg_timestamp = np.array([])

    for getdata in mydoc:
        diff1_tmp = np.array(getdata['Diff_1'])
        diff2_tmp = np.array(getdata['Diff_2'])
        diff3_tmp = np.array(getdata['Diff_3'])

        ecg_time = np.array(getdata['Ecg_time'])
        
        print('ecg_time:', ecg_time)
        
        diff1 = np.append(diff1, diff1_tmp)
        diff2 = np.append(diff2, diff2_tmp)
        diff3 = np.append(diff3, diff3_tmp)
        
        ecg_timestamp = np.append(ecg_timestamp, ecg_time)
    
    return diff1, diff2, diff3, ecg_timestamp


        
def read_SKH_3lead(folder, idx):
    '''
    用來讀取已從Mongo上抓下來並轉成mat檔的新光醫院收案資料
    '''
    if(folder == None):
        folder = 'SKH_Diff_3lead_clean'
    item = os.listdir(folder)
    ext = '.mat'
    filelist = []
    for i in range(len(item)):
        if(item[i].endswith(ext)):
            filelist.append(item[i])
            # print('[%d]' %i, item[i])
    item = filelist
    # for i in range(len(item)):
    for i in range(1):
        i = idx
        abspath = os.path._getfullpathname(folder) + '\\' + str(item[i])
        dataset = sio.loadmat(abspath)
        # diff1 = dataset['diff1']
        # diff2 = dataset['diff2']
        # diff3 = dataset['diff3']
    return dataset

def query_information(mydb, current_time):
    '''
    取得當下時間前後20秒內有上傳新資料的Patient
    '''
    status = 0
    lasttime_3leads = 0
    Patient_CodeID = ''
    mycol = mydb['user']
    # current_time = 1605078963
    start_time = current_time - 20
    end_time = current_time + 20
    myquery = {'lasttime_3lead': {'$gte': start_time,'$lte': end_time} }
    mydoc = mycol.find(myquery)
    for getdata in mydoc:
        Patient_CodeID = getdata['userId']
        lasttime_3leads = getdata['lasttime_3lead']
        status = getdata['Status']
        # lasttime_3leads = start
        
    return status, lasttime_3leads, Patient_CodeID

# if __name__ == '__main__':
#     item = os.listdir(SKH_3lead_clean_folder)
#     ext = 'mat'
#     filelist = []
#     for i in range(len(item)):
#         if(item[i].endswith(ext)):
#             filelist.append(item[i])
#             print('[%d]' %i, item[i])
#     item = filelist
#     for i in range(len(item)):
#         denoised = read_SKH_3lead(SKH_3lead_clean_folder, i)
#         ecg_3lead_insert(mydb, patient_ID, denoised, start + i)
    
#     status, lasttime_3lead, Patient_CodeID = query_information(mydb, 1606465600)