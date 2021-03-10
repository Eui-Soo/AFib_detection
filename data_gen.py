import numpy as np
import math
import os

Fs=50
ob_time=30
stride_time=10
stride_test_time=30
stride=stride_time*Fs
stride_test=stride_test_time*Fs
path_dir='./AFib'
AF_list=np.array(os.listdir(path_dir))
path_dir='./SR'
SR_list=np.array(os.listdir(path_dir))

test_ratio=0.2
N_AF_test=int(len(AF_list)*test_ratio)
N_SR_test=int(len(SR_list)*test_ratio)

test_idx=4 # 0 ~4
AF_index=np.arange(N_AF_test*test_idx,N_AF_test*(test_idx+1))
SR_index=np.arange(N_SR_test*test_idx,N_SR_test*(test_idx+1))
train_AF_list=np.delete(AF_list,AF_index)
train_SR_list=np.delete(SR_list,SR_index)
test_AF_list=AF_list[AF_index]
test_SR_list=SR_list[SR_index]
root_path=os.getcwd()

#train_AF_set=np.empty((1,Fs*ob_time),float)
os.chdir('./AFib')
for file_idx in range(len(train_AF_list)):
    data=np.loadtxt(train_AF_list[file_idx])
    N_set=int((len(data)-(Fs*ob_time-stride))/stride)
    tmp_set=np.zeros((N_set,Fs*ob_time))
    for n in range(N_set):
        tmp_set[n]=data[stride*(n):stride*(n)+Fs*ob_time]
        tmp_set[n]=tmp_set[n]-min(tmp_set[n])
        tmp_set[n]=tmp_set[n]/max(tmp_set[n])
    if file_idx==0:
        train_AF_set=tmp_set
    else:
        train_AF_set=np.append(train_AF_set,tmp_set,axis=0)
        
        
for file_idx in range(len(test_AF_list)):
    data=np.loadtxt(test_AF_list[file_idx])
    N_set=int((len(data)-(Fs*ob_time-stride_test))/stride_test)
    tmp_set=np.zeros((N_set,Fs*ob_time))
    for n in range(N_set):
        tmp_set[n]=data[stride_test*(n):stride_test*(n)+Fs*ob_time]
        tmp_set[n]=tmp_set[n]-min(tmp_set[n])
        tmp_set[n]=tmp_set[n]/max(tmp_set[n])
    if file_idx==0:
        test_AF_set=tmp_set
    else:
        test_AF_set=np.append(test_AF_set,tmp_set,axis=0)
        
os.chdir(root_path)
os.chdir('./SR')

for file_idx in range(len(train_SR_list)):
    data=np.loadtxt(train_SR_list[file_idx])
    N_set=int((len(data)-(Fs*ob_time-stride))/stride)
    tmp_set=np.zeros((N_set,Fs*ob_time))
    for n in range(N_set):
        tmp_set[n]=data[stride*(n):stride*(n)+Fs*ob_time]
        tmp_set[n]=tmp_set[n]-min(tmp_set[n])
        tmp_set[n]=tmp_set[n]/max(tmp_set[n])
    if file_idx==0:
        train_SR_set=tmp_set
    else:
        train_SR_set=np.append(train_SR_set,tmp_set,axis=0)
        
        
for file_idx in range(len(test_SR_list)):
    data=np.loadtxt(test_SR_list[file_idx])
    N_set=int((len(data)-(Fs*ob_time-stride_test))/stride_test)
    tmp_set=np.zeros((N_set,Fs*ob_time))
    for n in range(N_set):
        tmp_set[n]=data[stride_test*(n):stride_test*(n)+Fs*ob_time]
        tmp_set[n]=tmp_set[n]-min(tmp_set[n])
        tmp_set[n]=tmp_set[n]/max(tmp_set[n])
    if file_idx==0:
        test_SR_set=tmp_set
    else:
        test_SR_set=np.append(test_SR_set,tmp_set,axis=0)
os.chdir(root_path)

        
train_set=np.append(train_AF_set,train_SR_set,axis=0)
test_set=np.append(test_AF_set,test_SR_set,axis=0)
train_label=np.zeros((len(train_set),2))
train_label[0:len(train_AF_set)]=[1, 0]
train_label[len(train_AF_set):]=[0, 1]
test_label=np.zeros((len(test_set),2))
test_label[0:len(test_AF_set)]=[1, 0]
test_label[len(test_AF_set):]=[0, 1]
s='data_set_idx({0})_stride({1})'.format(test_idx, stride_time)
np.savez_compressed(s,a=train_set,b=train_label,c=test_set,d=test_label)
