# -*- coding: utf-8 -*-

import glob

import numpy as np
import cv2

from sklearn import svm

EPS=1.0e-40

# 存放训练数据和标签
train_data  = []    # 存放训练数据(图像特征数据构成的向量)
train_label = []    # 存放类型标签

# 存放验证数据和标签
test_data  = []
test_label = []

## 读入fname对应的深度图数据，经处理后得到特征向量
def get_feature_vector(fname):
    # 读取数据文件（对应深度图）
    img_src=np.genfromtxt(fname,delimiter=',').astype(np.float32)
    
    # 二值化
    _, img_src = cv2.threshold(img_src, 5, 255, cv2.THRESH_BINARY)
    
    # 去除孔洞和孤立点
    img_src_f = ~cv2.morphologyEx(img_src.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
    
    # 计算Hu不变矩
    h=cv2.HuMoments(cv2.moments(img_src_f)).ravel()
    
    if np.min(np.abs(h))<EPS: 
        return None
    else:
        return -np.sign(h)*np.log10(np.abs(h))
    
# 分析数据文件，得到特征数据和标签
print('Calculating feature vectors...')
cnt=0
for fname in glob.glob('demo\\*.csv'):
        label=fname[fname.rfind('\\')+1:fname.find('_')]
        print('fname:',fname,', label:',label)
        
        data=get_feature_vector(fname)
        if data is None: continue
        
        cnt+=1
        if cnt%2==0:
            # 训练集
            train_data.append(data)
            train_label.append(label)
        else:
            # 验证集
            test_data.append(data)
            test_label.append(label)

# 统计训练和验证数据量
print("Data set: %d for train / %d for test" % (len(train_data), len(test_data)))
label_list=list(set(train_label+test_label))# 得到标签(去除重复)
for label in label_list:
    train_cnt = (np.array(train_label) == label).sum()
    test_cnt  = (np.array( test_label) == label).sum()
    print("    %s: %d for train / %d for test" % (label,train_cnt,test_cnt))

# 训练SVM分类器
print("Training SVM classifier...")
classifier = svm.NuSVC(nu=0.1,gamma=0.5)    # 构造SVM分类器
classifier.fit(train_data, train_label)     # 训练

# 验证
print("Verify SVM classifier...")   
pre=classifier.predict(test_data)           # 使用训练好的分类器分类
acc = np.mean(pre == np.array(test_label))  # 计算正确率
print("accuracy: %.2f%%" % (acc*100))


