import tensorflow as tf
import numpy as np
import time
from scipy import io


Fs_data=400
Fs=50
ob_time=30
stride_time=10
data_idx=5
CART_add=True
num_models = 7


if data_idx==5:
    f='data_set_all_stride({0}).npz'.format(stride_time)
else:
    f='data_set_idx({0})_stride({1}).npz'.format(data_idx, stride_time)
data_load=np.load(f)
train_set=data_load['a']
train_label=data_load['b']

if CART_add==True:    
    mat_file = io.loadmat('ECG_denoising.mat')
    data_set=mat_file['ECG_buf']
    N_train_CART=204
    N_rep=10
    for nn in range(N_rep):
        for idx in range(N_train_CART):
            tmp_set=np.zeros((1,Fs*ob_time))
            L=data_set.shape
            tmp=data_set[idx]    
            tmp_set[0]=tmp[0:L[1]:int(Fs_data/Fs)]
            tmp_set=tmp_set-np.min(tmp_set)
            tmp_set=tmp_set/np.max(tmp_set)
            train_set=np.append(train_set,tmp_set,axis=0)       
        tmp_label=np.zeros((N_train_CART,2))
        tmp_label[:]=[0,1]
        train_label=np.append(train_label,tmp_label,axis=0)

N_train=len(train_set)
shuffle_idx=np.random.permutation(N_train)
train_set=train_set[shuffle_idx]
train_label=train_label[shuffle_idx]
train_set=train_set.reshape(N_train,1,Fs*ob_time,1)



def time_gen(S,Epoch,Total_Epoch):
    seconds=time.time()-S
    seconds=seconds*(Total_Epoch-Epoch-1)
    hour=seconds//3600
    minute=(seconds%3600)//60
    second=(seconds%3600)%60
    if hour > 0:
        time_result="%d hour %d min %d sec" % (hour,minute,second)
    elif hour == 0 and minute>0:
        time_result="%d min %d sec" % (minute,second)
    elif hour == 0 and minute==0:
        time_result="%d sec" % second
    return time_result


learning_rate = 0.001
epochs = 10
batch_size = 200
train_len=len(train_set)
total_batch = int(train_len/batch_size)
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
            
            W2 = tf.Variable(tf.random_normal([1,9,L1_unit,L2_unit], stddev = 0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding = 'SAME')
            self.L2 = tf.nn.relu(L2)
            L2 = tf.nn.dropout(self.L2, self.keep_prob2)
            L2 = tf.nn.max_pool(L2, ksize=[1,1,4,1], strides=[1,1,4,1], padding='SAME')
            
            W3 = tf.Variable(tf.random_normal([1,9,L2_unit,L3_unit], stddev = 0.01))
            L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding = 'SAME')
            self.L3 = tf.nn.relu(L3)
            L3 = tf.nn.dropout(self.L3, self.keep_prob3)
            L3 = tf.nn.max_pool(L3, ksize=[1,1,4,1], strides=[1,1,4,1], padding='SAME')
            
            Wf = tf.Variable(tf.random_normal([1*24*L3_unit,2], stddev = 0.01))
            L = tf.reshape(L3,[-1,1*24*L3_unit])
            self.model = tf.matmul(L,Wf)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model, labels = self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

    def train(self, x_data, y_data, keep_prop1=0.7,keep_prop2=0.8,keep_prop3=0.9):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data,
            self.Y: y_data,
            self.keep_prob1: keep_prop1,
            self.keep_prob2: keep_prop2,
            self.keep_prob3: keep_prop3})


sess =tf.Session()
models = []
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))
sess.run(tf.global_variables_initializer())

print('Learning Started!')
saver = tf.train.Saver()
if data_idx==5:
    save_file = './model/1D_CNN_model_dataset(all)_CART_{0}_model({1}).ckpt' .format(CART_add,num_models)
else:
    save_file = './model/1D_CNN_model_dataset({0})_CART_{1}_model({2}).ckpt' .format(data_idx,CART_add,num_models)
        
for epoch in range(epochs):
    S=time.time()
    avg_cost = np.zeros(len(models))
    for i in range(total_batch):
        batch_x = train_set[i*batch_size:(i+1)*batch_size]
        batch_y = train_label[i*batch_size:(i+1)*batch_size]
        for m_idx, m in enumerate(models):
            cost_val,_ = m.train(batch_x, batch_y)
            avg_cost[m_idx] += cost_val / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost)
    print('Time remaining : '+ time_gen(S,epoch,epochs))
saver.save(sess, save_file)
print('Learning Finished')





