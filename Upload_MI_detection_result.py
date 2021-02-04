# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 08:57:02 2021

@author: lab711 Alex Chen
Goal: Upload the result of detection MI
"""
import pymongo
import numpy as np
from scipy import signal
import time
import warnings
warnings.filterwarnings("ignore")

from keras.models import load_model
from keras.optimizers import Adam

from MongoUse import *
#from solve_cudnn_error import *



def check_length(input_data, second, freq, tolerance = 5, DataL=2048):
    '''
    會掉封包所以要補足資料長度
    最少要拿到長度1275的資料才判定有拿到完整5秒
    '''
    data_len = len(input_data)
    required_len = int(second * freq) # 5* 256 = 1280
    if(data_len < required_len):
        if(data_len < required_len - tolerance):
            return 0
        else:
            input_data = input_data.tolist()
            sub = required_len - data_len
            for i in range(sub): # 少幾個值就補多少個0
                input_data.append(0)
            input_data = np.array(input_data)
    elif(data_len > required_len): # 以防萬一拿到超過1280個sampless
        input_data = input_data.tolist()
        sub = data_len - required_len
        for i in range(sub): # 多幾個值就補多少個0
            input_data.pop(-1)
        input_data = np.array(input_data)
        
    else:
        print("Correct data length! (%d)", required_len)

    output = input_data.flatten()
    output = signal.resample(output,DataL)
    output = np.array(output).reshape(1,DataL,1)
    return output

def acc(predict, answer):
    output_clear = np.zeros(predict.shape)
    N = predict.shape[0]
    output_clear = predict>0.7
    for i in range(N):
        if any(predict[i,1:]):
            predict[i,0]=0
        else:
            predict[i,0]=1
    
    acc = 0    
    for i in range(N):
        if sum(abs(output_clear[i,:]-answer[i,:]))==0:
            acc = acc+1
    acc = acc/N*100
    print("accurarcy = %3.3f%%" % acc)
    return N,output_clear


def statistic(conf_mat, N):
    epsilon3 = np.ones(3)*1e-10
    print("\t\tNormal\tAMI\tIMI")
    accu = (conf_mat[:,0,0]+conf_mat[:,1,1])/N
    print("accurarcy=\t%3.3f\t%3.3f\t%3.3f" % tuple(accu))
    prec = conf_mat[:,1,1]/(conf_mat[:,1,1]+conf_mat[:,0,1]+epsilon3)
    print("precision=\t%3.3f\t%3.3f\t%3.3f" % tuple(prec))
    recall = conf_mat[:,1,1]/(conf_mat[:,1,1]+conf_mat[:,1,0]+epsilon3)
    print("recall=\t\t%3.3f\t%3.3f\t%3.3f" % tuple(recall))
    F1_score = 2/(1/(prec+epsilon3)+1/(recall+epsilon3))
    print("F1 score=\t%3.3f\t%3.3f\t%3.3f\n" % tuple(F1_score))
    
def stft(data, fs, pt, step):
    data = np.array(data)   
    operN = (len(data)-pt)//step+1 # 做FFT次數
    output = np.zeros([pt//2,operN]) # 輸出變數
    for i in range(operN):
        data_fft = np.abs(fft(data[(pt//2)*i:(pt//2)*(i+1)-1],fs)[0:pt//2])/fs # 做FFT
        data_fft = data_fft # 取0以及正頻
        output[:,i] = data_fft
    #plt.figure()
    output = np.matrix(output)
    #plt.pcolormesh(output)
    return output


if __name__ == '__main__':
    Mode2D = 0
    last_status = 0
    sec_len = 5     # 5 sec
    sample_rate = 256
    feature_num = sec_len * sample_rate
    counter = 0
    last_update_time = 0 # 紀錄上次上傳5秒資料的開始Timestamp
    
    # MongoDB information
    myclient = pymongo.MongoClient("mongodb://192.168.25.22:27017/")
    mydb = myclient['ecg']
    mycol = mydb['ecg_realtime_data']
    current_time = int(time.time())
    # current_time = 1605078963
    print("Current Time:", current_time)
    if not Mode2D:
        model = load_model("model2.hdf5", compile=False)
    else:
        model = load_model("model2d.hdf5", compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    print("=============== Model loading completed! ============")
        
    while True:
        # 取得最近有上傳3導極資料的使用者資訊
        current_time = int(time.time())
        # current_time = 1605078963
        user_status, last_time_3lead, Patient_CodeID = query_information(mydb, current_time)
        print("User status check!")
        print(user_status)
        time.sleep(1)
        
        while(user_status == 0):
            current_time = int(time.time())
            user_status, last_time_3lead, Patient_CodeID = query_information(mydb, current_time)
            print("Last time 3 lead update:", last_time_3lead)
            print("Current time:", current_time)
            print("Patient ID:", Patient_CodeID)
            print("Status:", user_status)
            time.sleep(1)
            
        while(user_status == 1):
            if(last_status == 0): # status: 0 → 1 start running
                # 當正在轉的情況下仍要：
                # 1. 持續check user table的最新上傳三導極時間
                # 2. 持續更新當下timestamp
                current_time = int(time.time())
                user_status, last_time_3lead, Patient_CodeID = query_information(mydb, current_time)
                start_time = last_time_3lead - (sec_len - 1) # 拿到的時間是該5秒的最後一秒timestamp
                print("Transfer start timestamp:", start_time)
                print("last_time_3lead:", last_time_3lead)
                print("\n")
            
                # print("！！！", start_time - last_update_time)
                # if(start_time < last_time_3lead):
                if(start_time - last_update_time >= sec_len):
                    
                    # start_time = last_time_3lead
                    # start_time = 1605078816
                    end_time = start_time + sec_len
                    diff1, diff2, diff3, timestamp_list = collect_mongo_data(mycol, start_time, end_time, Patient_CodeID)
                    print("Data collected.")
                    

                    if(len(timestamp_list) < sec_len):
                        print("Data length < 5 sec")
                        start_time += 1 # 再往下一秒找連續的5秒
                        time.sleep(1)
                        counter += 1
                        if(counter >= sec_len): # 等超過5秒還是沒有足夠資料
                            user_status, last_time_3lead, Patient_CodeID = query_information(mydb, current_time)
                            print("User status check!")
                            print(user_status)
                            continue
                    else:
                        # Start running
                        now_time = time.time()
                        start_date = time.ctime(now_time)
                        print("Start Date:", start_date)
                        print("Start Timestamp: ", start_time)
                        end_date = time.ctime(now_time + sec_len)
                        print("End Date:", end_date)
                        
                        # 確定有資料可以轉後
                        # 若有掉封包則補足長度到1280，並順便reshape成model預設的input大小
                        if not Mode2D:
                            diff1 = check_length(diff1, sec_len, sample_rate)
                            diff2 = check_length(diff2, sec_len, sample_rate)
                            diff3 = check_length(diff3, sec_len, sample_rate)
                            print("Data length modification finish!")
                            
                            # Start MI detection
                            diff1 = np.append(diff1,diff1,axis=1).reshape(1,4096,1)
                            diff2 = np.append(diff2,diff2,axis=1).reshape(1,4096,1)
                            diff3 = np.append(diff3,diff3,axis=1).reshape(1,4096,1)
                            x = np.append(diff1,diff2,axis=2)
                            x = np.append(x,diff3,axis=2)
                        else:
                            diff1 = check_length(diff1, sec_len, sample_rate, DataL=1250)
                            diff2 = check_length(diff2, sec_len, sample_rate, DataL=1250)
                            diff3 = check_length(diff3, sec_len, sample_rate, DataL=1250)
                            print("Data length modification finish!")
                            
                            # Start MI detection
                            diff1 = np.append(diff1,diff1,axis=1).reshape(1,2500,1)
                            diff2 = np.append(diff2,diff2,axis=1).reshape(1,2500,1)
                            diff3 = np.append(diff3,diff3,axis=1).reshape(1,2500,1)

                            # STFT
                            Fs = 250
                            x = np.zeros([1,Fs//2,19,3])    # 1筆 125
                            
                            x[:,:,:,0] =  stft(diff1, fs=Fs, pt=Fs, step=Fs//2)
                            x[:,:,:,1] =  stft(diff2, fs=Fs, pt=Fs, step=Fs//2)
                            x[:,:,:,2] =  stft(diff3, fs=Fs, pt=Fs, step=Fs//2)
                        
                        print("Start detecting...", end="")    
                        start = time.time()
                        y_score = model.predict(x, batch_size=1, verbose=0)
                        cost = time.time()-start
                        print("---------------------")
                        print("Cost time: %1.2f s" % cost)
                        # 判斷結果
                        result = y_score>0.7
                        if result[0,0]:
                            result = 1
                            mi_type = "IMI"
                            print("Result：MI (Myocardia infarction)")
                            print("Likelihood：%2.2f" % y_score[0,0])
                            print("Type:IMI (Inferior myocardia infarction)\n")
                        elif result[0,1]:
                            result = 1
                            mi_type = "AMI"
                            print("Result：MI (Myocardia infarction)")
                            print("Likelihood：%2.2f" % y_score[0,1])
                            print("Type：AMI (Anterior myocardia infarction)\n")
                        else:
                            result = 0
                            mi_type = None
                            print("Result:Normal\n")
                            print("Detection done!")
                        
                        
                        # 檢測結果上傳到MongoDB
                        upload_dict = {'predict_result':result, 'mi_type':mi_type}
                        mi_result_insert(mydb, Patient_CodeID, upload_dict, start_time)
                        
                        last_update_time = start_time
                        print("Upload MI prediction result finished.")
                        print("Upload Timestamp:", last_update_time) # 拿到的資料開頭Timestamp = 上傳回去的資料開頭Timestamp
                        print("==========UPLOAD FINISHED!!!========\n")
                        time.sleep(2)
                else: #當start_time - last_update_time < 5秒時
                    print("Time difference shorter than 5 sec:", start_time - last_update_time)
                    time.sleep(1)

