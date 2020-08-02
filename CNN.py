# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:59:22 2018

@author: lj
"""
import numpy as np

# 导入数据    # Import Data
def load_data(path, feature_num=2):
    '''导入数据  Import Data
    input:  path(string)文件的存储位置  File storage location
            feature_num(int)特征的个数  Number of features
    output: data(array)特征  feature
    '''
    f = open(path)  # 打开文件  # open a file
    data = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        data_tmp = []
        if len(lines) != feature_num:  # 判断特征的个数是否正确 # Determine whether the number of features is correct
            continue
        for i in range(feature_num):
            data_tmp.append(float(lines[i]))
        data.append(data_tmp)
    f.close()  # 关闭文件 # Close file
    return data

def sigmoid(x):#激活函数  #Activation function
    return 1/(1+np.exp(-x))

def normalization(M):
    """对行向量进行归一化   Normalize the row vector
    :param M:行向量   Row vector：【dim=len(M)】
    :return: 归一化后的行向量M   Normalized row vector M
    """
    M=M/np.sqrt(np.dot(M,M.T))
    return M

def normalization_all(N):
    """对矩阵进行归一化 Normalize the matrix
    :param N: 矩阵   matrix：【m,n】
    :return: 归一化后的矩阵(Normalized matrix)M_all:【m,n】
    """
    M_all=[]
    for i in range(len(N)):
        K=normalization(N[i])
        M_all.append(K)
    return M_all

class competitive_network(object):
    def __init__(self,x_dim,output_num,a):
        '''类参数初始化  Class parameter initialization
        '''
        W = np.random.rand(output_num,x_dim)
        self.W = normalization_all(W)
        self.a = a ## 权值更新参数  Weight update parameter
        
    def forward_propagation(self,x):
        '''前向传播  Forward propagation
        input:self(object):类参数  Class parameter
              x(mat):一个训练样本  Training samples
        output:argmax(int):被激活的权重向量指针  Activated weight vector pointer
        '''
        z_layer=np.dot(self.W,x.T) ##矩阵相乘  Matrix multiplication
        a_layer=sigmoid(z_layer) 
        argmax= np.argmax(a_layer)
        return argmax
    
    def back_propagation(self,argmax,x):
        '''反向传播调整权重  Backpropagation adjustment weight
        input:argmax(int):被激活的权重向量指针 Activated weight vector pointer
              x(mat):一个训练样本  Training samples
        '''
        self.W[argmax] = self.a * (x - self.W[argmax])
        self.W[argmax]=normalization(self.W[argmax])
        self.a-=self.decay
    
    def train(self,X,num_iter):
        '''模型训练  Model training
        input:X(mat):全部训练样本  all training samples
              num_iter(int):迭代次数  the number of iterations
        '''
        X=np.array(X)
        self.decay=self.a / num_iter
        for item in range(num_iter):
            for i in range(X.shape[0]):
                argmax=self.forward_propagation(X[i])
                self.back_propagation(argmax,X[i])
            
    def prediction(self,X_test):
        '''预测样本的类别  Class of prediction sample
        input:self(object):类 class
              X_test(mat):测试样本 test sample
        output:predict_class(list):样本类别  sample category
        '''
        sample_num = np.shape(X_test)[0]
        predict_results = []
        for i in range(sample_num):
            predict_result = self.forward_propagation(X_test[i])
            predict_results.append(predict_result)
        return predict_results
        


if __name__ == '__main__':
    print('---------------------1.Load Data---------------------')
    data = load_data('data')
    dataMat = np.mat(data)
    print('------------------2.Parameters Seting----------------')
    num_iter = 1000
    x_dim = np.shape(dataMat)[1]
    output_num = 2
    a = 0.3
    print('-------------------3.Model Train---------------------')
    cnn = competitive_network(x_dim,output_num,a)
    cnn.train(dataMat,num_iter)
    print('-------------------4.Prediction----------------------')
    predict_results = cnn.prediction(dataMat)
    
    
