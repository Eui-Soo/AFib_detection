#import tensorflow as tf
import numpy as np
import time
import math
import os

Fs=50
ob_time=30
stride_time=10
stride=stride_time*Fs
path_dir='./AFib'
train_AF_list=np.array(os.listdir(path_dir))
path_dir='./SR'
train_SR_list=np.array(os.listdir(path_dir))

root_path=os.getcwd()

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
        
os.chdir(root_path)

        
train_set=np.append(train_AF_set,train_SR_set,axis=0)
train_label=np.zeros((len(train_set),2))
train_label[0:len(train_AF_set)]=[1, 0]
train_label[len(train_AF_set):]=[0, 1]

s='data_set_all_stride({0})'.format(stride_time)
np.savez_compressed(s,a=train_set,b=train_label)
