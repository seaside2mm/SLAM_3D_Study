
"""
深蓝学院。文件功能： 实现 K-Means 算法

将输入的N个数据点，分为N个类
1.随机选取K个中心点
2.E-Step(expectation)：N个点、K个中心，求N个点到K个中心的nearest-neighbor
3.M-Step(maximization)：更新中心点的位置，把属于同一个类的数据点求一个均值，作为这个类的中心值
4.不断重复2、3步
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from result_set import KNNResultSet, RadiusNNResultSet
import  kdtree as kdtree

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    # 进行中心点的确定
    def fit(self, data):
        #TODO 
        #step1 随机选取 K个数据点 作为聚类的中心
        self.centers_ = data[random.sample(range(data.shape[0]),self.k_)]    #random.sample(list,num)
        old_centers = np.copy(self.centers_)  #存储old_centers
        
        #step2 E-Step(expectation)：N个点、K个中心，求N个点到K个中心的nearest-neighbor
        #kd-tree config
        leaf_size = 1
        k = 1  # 结果每个点选取属于自己的类中心
        for _ in range(self.max_iter_):
            labels = [[] for i in range(self.k_)]        #用于分类所有数据点
            root = kdtree.kdtree_construction(self.centers_ , leaf_size=leaf_size)    #对中心点进行构建kd-tree
            for i in range(data.shape[0]):       #对每一个点在4个中心点中进行 1-NN的搜索
                result_set = KNNResultSet(capacity=k)
                query =  data[i]
                kdtree.kdtree_knn_search(root, self.centers_, result_set, query)     #返回对应中心点的索引
                # labels[result_set.output_index].append(data[i])
                #print(result_set)
                output_index = result_set.knn_output_index()[0]                 #获取最邻近点的索引
                labels[output_index].append(data[i])             #将点放入类中
            
            #step3 M-Step(maximization)：更新中心点的位置，把属于同一个类的数据点求一个均值，作为这个类的中心值
            for i in range(self.k_):     #求K类里，每个类的的中心点
                points = np.array(labels[i])
                self.centers_[i] = points.mean(axis=0)       #取点的均值，作为新的聚类中心
                # print(points)
                # print(self.centers_[i])
            if np.sum(np.abs(self.centers_ - old_centers)) < self.tolerance_ * self.k_:  # 如果前后聚类中心的距离相差小于self.tolerance_ * self.k_ 输出
                break
            old_centers = np.copy(self.centers_)     #保存旧中心点
        self.fitted = True
        Point_Show(self.centers_)


    #计算每个点的类别
    def predict(self, p_datas):
        result = []
        # TODO:作业2
        if not self.fitted:
            print('Unfitter. ')
            return result
        for point in p_datas:
            diff = np.linalg.norm(self.centers_ - point, axis=1)     #使用二范数求解每个点对新的聚类中心的距离
            result.append(np.argmin(diff))                           #返回离该点距离最小的聚类中心，标记rnk = 1
    return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)

