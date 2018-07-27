#!/usr/bin/env python
# -*-coding:utf-8 -*-
# ** author:toby
# ** description: more function
# ** run python: python xxxx.py param1 param2
# ********************
import numpy as np
import os
import logging
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='myapp.log',
                filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s  %(filename)s[line:%(lineno)d] %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
def load_data(path):
    '''
    处理一个文件夹内数据的方法
    :param path: train
    :return:
    '''
    label = []
    data = []
    file_name = list(os.listdir(path))
    logging.info('处理数据....')
    for text in  file_name :
        file_path = os.path.join(path,text)
        with open(file_path,'r') as fi:
            cnt = fi.readlines()
        res = [i.strip().split('\n')[0] for i in cnt]
        res = ''.join(res)
        res = list(map(int,res))
        data.append(res)
        label.append(file_path.strip().split('/')[-1].split('_')[0])
    return np.array(data),np.array(label,dtype =np.int).reshape(-1,1)

def sigmoid(x):
    new_x = x-0    # 非常重要，不能随便搞
    return 1.0/(1+np.exp(-new_x))

def gradAscent(data, y,alpha, max_cycles):
    W = np.ones((data.shape[1],1))
    for i in range(max_cycles):
        logging.info('迭代计算梯度,cycle =={}'.format(i))
        h = sigmoid(np.matmul(data,W))
        error = y - h
        W =W +alpha*np.matmul(data.transpose(),error)
    return W

def predict(test,W,threshhold=  0.5):
    logging.info('预测结果中')
    score = np.array(list(map(sigmoid,(np.matmul(test,W)))))
    class_ = (score >= threshhold).astype(np.int)
    return class_.reshape(-1,1)

def metrics(predict:np.array, label:np.array):
    logging.info('进行度量')
    tp = sum(predict[np.where(label==1)])
    fp = sum(predict[np.where(label==0)])
    tn = sum(predict[np.where(label==0)]==0)
    fn = sum(predict[np.where(label==1)]==0)
    pre= tp/(tp+fp)
    recall = tp/(tp+fn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    return pre , recall ,acc




if __name__=="__main__":
    train, train_label = load_data('./train/')
    test,test_label=load_data('./test/')
    weight = gradAscent(train,train_label,0.07,10)
    pred = predict(test,weight,0.5)
    print(metrics(pred ,test_label))

