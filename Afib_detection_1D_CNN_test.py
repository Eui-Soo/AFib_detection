import tensorflow as tf
import numpy as np
import time
from scipy import io
from collections import Counter
Fs_data=400
Fs=50
ob_time=30
stride_time=10
data_idx=0
CART_add=True
test_mode=1 # 0: physiolab, 1: CART data
num_models = 1

if test_mode==0:
    f='data_set_idx({0})_stride({1}).npz'.format(data_idx, stride_time)
    data_load=np.load(f)
    test_set=data_load['c']
    test_label=data_load['d']
    N_test=len(test_set)
elif test_mode==1:
    mat_file = io.loadmat('ECG_denoising.mat')
    data_set=mat_file['ECG_buf']
    if CART_add == True:
        N_train_CART=100
        for idx in range(N_train_CART,204):
            tmp_set=np.zeros((1,Fs*ob_time))
            L=data_set.shape
            tmp=data_set[idx]    
            tmp_set[0]=tmp[0:L[1]:int(Fs_data/Fs)]
            tmp_set=tmp_set-np.min(tmp_set)
            tmp_set=tmp_set/np.max(tmp_set)
            if idx==N_train_CART:
                test_set=tmp_set
            else:
                test_set=np.append(test_set,tmp_set,axis=0)
            N_test=L[0]-N_train_CART
        test_label=np.zeros((N_test,2))
        test_label[:]=[0,1]
    elif CART_add==False:
        for idx in range(0,204):
            tmp_set=np.zeros((1,Fs*ob_time))
            L=data_set.shape
            tmp=data_set[idx]    
            tmp_set[0]=tmp[0:L[1]:int(Fs_data/Fs)]
            tmp_set=tmp_set-np.min(tmp_set)
            tmp_set=tmp_set/np.max(tmp_set)
            if idx==0:
                test_set=tmp_set
            else:
                test_set=np.append(test_set,tmp_set,axis=0)
            N_test=len(test_set)
        test_label=np.zeros((N_test,2))
        test_label[:]=[0,1]
test_set=test_set.reshape(N_test,1,Fs*ob_time,1)


L1_unit=128
L2_unit=256
L3_unit=512

class Model:
    def __init__(self, sess, name):
        self.sess=sess
        self.name=name
        self._build_net()
        
    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32,shape=[None,1,Fs*ob_time,1])
            self.Y = tf.placeholder(tf.float32, shape=[None,2])
            self.keep_prob1 = self.keep_prob2 = self.keep_prob3 = tf.placeholder(tf.float32)
            W1 = tf.Variable(tf.random_normal([1,9,1,L1_unit], stddev = 0.01))
            L1 = tf.nn.conv2d(self.X, W1, strides=[1,1,1,1], padding = 'SAME')
            self.L1 = tf.nn.relu(L1)
            L1 = tf.nn.dropout(self.L1, self.keep_prob1)
            L1 = tf.nn.max_pool(L1, ksize=[1,1,4,1], strides=[1,1,4,1], padding='SAME')
            ###(1,750)
            W2 = tf.Variable(tf.random_normal([1,9,L1_unit,L2_unit], stddev = 0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding = 'SAME')
            self.L2 = tf.nn.relu(L2)
            L2 = tf.nn.dropout(self.L2, self.keep_prob2)
            L2 = tf.nn.max_pool(L2, ksize=[1,1,4,1], strides=[1,1,4,1], padding='SAME')
            #######(1,375)
            W3 = tf.Variable(tf.random_normal([1,9,L2_unit,L3_unit], stddev = 0.01))
            L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding = 'SAME')
            self.L3 = tf.nn.relu(L3)
            L3 = tf.nn.dropout(self.L3, self.keep_prob3)
            L3 = tf.nn.max_pool(L3, ksize=[1,1,4,1], strides=[1,1,4,1], padding='SAME')
            
            Wf = tf.Variable(tf.random_normal([1*24*L3_unit,2], stddev = 0.01))
            L = tf.reshape(L3,[-1,1*24*L3_unit])
            self.model = tf.matmul(L,Wf)
        x=tf.argmax(self.model, 1)
        y=tf.argmax(self.Y, 1)
        is_correct = tf.equal(x, y)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    def predict(self, x_test,y_test, keep_prop1=1.0,keep_prop2=1.0,keep_prop3=1.0):
        return self.sess.run([self.accuracy,self.model], feed_dict={self.X: x_test,
                                                                    self.Y: y_test,
                                                                    self.keep_prob1: keep_prop1,
                                                                    self.keep_prob2: keep_prop2,
                                                                    self.keep_prob3: keep_prop3})



sess =tf.Session()
models = []

for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))
if data_idx==5:
    save_file = './model/1D_CNN_model_dataset(all)_CART_{0}_model({1}).ckpt' .format(CART_add,num_models)
else:
    save_file = './model/1D_CNN_model_dataset({0})_CART_{1}_model({2}).ckpt' .format(data_idx,CART_add,num_models)
saver = tf.train.Saver()
saver.restore(sess, save_file)

test_batch=100
test_len=len(test_set)
total_batch=int(np.ceil(test_len/test_batch))
predictions = np.zeros([test_len, 2])
each_predictions = np.zeros([num_models,test_len, 2])
for i in range(total_batch):
    batch_x = test_set[i*test_batch:(i+1)*test_batch]
    batch_y = test_label[i*test_batch:(i+1)*test_batch]
    for m_idx, m in enumerate(models):
        a,p = m.predict(batch_x,batch_y)
        predictions[i*test_batch:(i+1)*test_batch] += p[:]
        each_predictions[m_idx][i*test_batch:(i+1)*test_batch][:]=p[:]




if test_mode==0:
    for idx in range(num_models):
        buf=np.argmax(test_label,axis=1)-np.argmax(each_predictions[idx],axis=1)
        buf_label=np.argmax(test_label,axis=1)
        count1 = Counter(buf)
        count2 = Counter(buf_label)
        label_AF_pred_SR=count1[-1]/count2[0] 
        label_SR_pred_AF=count1[1]/count2[1] 
        print('network ',(idx+1))
        print('Acc. : ',count1[0]/test_len)         
        print('sensitivity : ',1-label_AF_pred_SR) 
        print('specificity : ',1-label_SR_pred_AF)  
        print(' ')
    if num_models!=1:
        buf=np.argmax(test_label,axis=1)-np.argmax(predictions,axis=1)
        buf_label=np.argmax(test_label,axis=1)
        count1 = Counter(buf)
        count2 = Counter(buf_label)
        label_AF_pred_SR=count1[-1]/count2[0] 
        label_SR_pred_AF=count1[1]/count2[1]    
        print('ensemble method')
        print('Acc. : ',count1[0]/test_len)         
        print('sensitivity : ',1-label_AF_pred_SR) 
        print('specificity : ',1-label_SR_pred_AF)
elif test_mode==1:
    for idx in range(num_models):
        buf=np.argmax(test_label,axis=1)-np.argmax(each_predictions[idx],axis=1)
        buf_label=np.argmax(test_label,axis=1)
        count1 = Counter(buf)
        count2 = Counter(buf_label)
        label_SR_pred_AF=count1[1]/count2[1] 
        print('network ',(idx+1))
        print('Acc. : ',count1[0]/test_len)          
        print('specificity : ',1-label_SR_pred_AF)  
        print(' ')
    if num_models!=1:
        buf=np.argmax(test_label,axis=1)-np.argmax(predictions,axis=1)
        buf_label=np.argmax(test_label,axis=1)
        count1 = Counter(buf)
        count2 = Counter(buf_label)
        label_SR_pred_AF=count1[1]/count2[1]    
        print('ensemble method')
        print('Acc. : ',count1[0]/test_len)         
        print('specificity : ',1-label_SR_pred_AF)






