# 文件功能：实现 Spectral 谱聚类 算法
# DBSCAN-使用距离矩阵法-编写流程
# step1: 建立数据集中每个点两两点的距离矩阵，距离矩阵为对角矩阵，对角线为0
# step2: 通过 eps半径 Minpts:最小样本点数 的判定，找出所有核心点
# step3: 进行聚类

from numpy import *
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import  time

plt.style.use('seaborn')


# matplotlib显示点云函数
def Point_Show(point,color):
    x = []
    y = []
    point = np.asarray(point)
    for i in range(len(point)):
        x.append(point[i][0])
        y.append(point[i][1])
    plt.scatter(x, y,color=color)


#构建距离矩阵
def my_distance_Marix(data):
    S = np.zeros((len(data), len(data)))  # 初始化 关系矩阵 w 为 n*n的矩阵
    # step1 建立关系矩阵， 每个节点都有连线，权重为距离的倒数
    for i in range(len(data)):  # i:行
        for j in range(len(data)):  # j:列
                S[i][j] = np.linalg.norm(data[i] - data[j])  # 二范数计算两个点直接的距离，两个点之间的权重为之间距离的倒数
    return S

# @profile
def DBSCAN(data, eps, Minpts):
    """
    基于密度的点云聚类
    :param d_bbox: 点与点之间的距离矩阵
    :param eps:  最大搜索直径阈值
    :param Minpts:  最小包含其他对象数量阈值
    :return: 返回聚类结果，是一个嵌套列表,每个子列表就是这个区域的对象的序号
    """
    # 先求距离
    d_bbox = my_distance_Marix(data)
    #step1 初始化核心对象集合T,聚类个数k,聚类集合C, 未访问集合P
    T = set()    #set 集合
    k = 0        #类初始化
    C = []       #聚类集合
    unvisited = set(range(d_bbox.shape[0]))   #初始化未访问集合
    #step2 通过判断，两点得 distance 找出所有核心点
    for d in range(d_bbox.shape[0]):
        if np.sum(d_bbox[d,:] <= eps) >= Minpts:     #临近点数 > min_sample,加入核心点
            T.add(d)    #最初得核心点
    #step3 聚类
    while len(T):     #visited core ，until all core points were visited
        unvisited_old = unvisited     #更新为访问集合
        core = list(T)[np.random.randint(0,len(T))]    #从 核心点集T 中随机选取一个 核心点core
        unvisited = unvisited - set([core])      #把核心点标记为 visited,从 unvisited 集合中剔除
        visited = []
        visited.append(core)

        while len(visited):
            new_core = visited[0]
            new_core_nearest = np.where(d_bbox[new_core,:] <= eps)[0]   #获取new_core 得邻近点
            if len(new_core_nearest) >= Minpts:
                S = unvisited & set(new_core_nearest)    #当前 核心对象的nearest 与 unvisited 的交集
                visited +=  (list(S))                     #对该new core所能辐射的点，再做检测
                unvisited = unvisited - S          #unvisited 剔除已 visited 的点
            visited.remove(new_core)                     #new core 已做检测，去掉new core

        k += 1   #类个数
        cluster = unvisited_old -  unvisited    #原有的 unvisited # 和去掉了 该核心对象的密度可达对象的visited就是该类的所有对象
        T = T - cluster  #去掉该类对象里面包含的核心对象
        C.append(cluster)  #把对象加入列表
    print("生成的聚类个数：%d" %k)
    return C,k
# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    #plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    begin_t = time.time()
    index,k = DBSCAN(X,eps=0.5,Minpts=15)
    dbscan_time = time.time() - begin_t
    print("dbscan time:%f" %dbscan_time)
    cluster = [[] for i in range(k)]
    for i in range(k):
        cluster[i] = [X[j] for j in index[i]]

    Point_Show(cluster[0],color="red")
    Point_Show(cluster[1], color="orange")
    Point_Show(cluster[2],color="blue")
    plt.show()
    # dbscan.fit(X)
    # K = 3
    # spectral.fit(X)
    # cat = spectral.predict(X)
    # print(cat)
    # cluster =[[] for i in range(K)]
    # for i in range(len(X)):
    #     if cat[i] == 0:
    #         cluster[0].append(X[i])
    #     elif cat[i] == 1:
    #         cluster[1].append(X[i])
    #     elif cat[i] == 2:
    #         cluster[2].append(X[i])
    # Point_Show(cluster[0],color="red")
    # Point_Show(cluster[1], color="orange")
    # Point_Show(cluster[2],color="blue")
    # plt.show()