# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

# matplotlib显示点云函数
def Point_Cloud_Show(points):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], cmap='spectral', s=2, linewidths=0, alpha=1, marker=".")
    plt.title('Point Cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


# 二维点云显示函数
def Point_Show(pca_point_cloud):
    x = []
    y = []
    pca_point_cloud = np.asarray(pca_point_cloud)
    for i in range(10000):
        x.append(pca_point_cloud[i][0])
        y.append(pca_point_cloud[i][1])
    plt.scatter(x, y)
    plt.show()


# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # TODO：作业1
    average_data = np.mean(data,axis=0)       #求 NX3 向量的均值
    decentration_matrix = data - average_data   #去中心化
    H = np.dot(decentration_matrix.T,decentration_matrix)  #求解协方差矩阵 H
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)    # SVD求解特征值、特征向量
    
    if sort:
        sort = eigenvalues.argsort()[::-1]      #降序排列
        eigenvalues = eigenvalues[sort]         #索引
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    ############################## 读取数据 ##############################
    point_cloud_raw = np.genfromtxt(r"/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/05.3D特征提取与参数测量/CH5/airplane_0001.txt", delimiter=",")  #为 xyz的 N*3矩阵
    
    point_cloud_raw = DataFrame(point_cloud_raw[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud  = np.array(point_cloud_raw)
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    #o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    ############################## 用PCA分析点云主方向  ##############################
    w, v = PCA(point_cloud_raw)        # w为特征值 v为主方向
    point_cloud_vector1 = v[:, 0]   #点云主方向对应的向量，第一主成分
    point_cloud_vector2 = v[:, 1]  # 点云主方向对应的向量，第二主成分
    point_cloud_vector = v[:,0:2]  # 点云主方向与次方向
    print('the main orientation of this pointcloud is: ', point_cloud_vector1)
    print('the main orientation of this pointcloud is: ', point_cloud_vector2)

    ##画点：原点、第一主成分、第二主成分
    point = [[0,0,0],point_cloud_vector1,point_cloud_vector2]  
    lines = [[0,1],[0,2]]      #画出三点之间两两连线
    colors = [[1,0,0],[0,0,0]]
    #构造open3d中的LineSet对象，用于主成分显示
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    #o3d.visualization.draw_geometries([point_cloud_o3d,line_set]) # 显示原始点云和PCA后的连线
    
    ############################## 降维与升维  ##############################
    #将原数据进行降维度处理，主成分的转置 dot 原数据
    point_cloud_encode = (np.dot(point_cloud_vector.T,point_cloud_raw.T)).T   
    #Point_Show(point_cloud_encode)
    
    #使用主方向进行升维
    point_cloud_decode = (np.dot(point_cloud_vector,point_cloud_encode.T)).T
    #Point_Cloud_Show(point_cloud_decode)
    
    ############################## 法向量估计  ##############################
    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)  #将原始点云数据输入到KD,进行近邻取点
    normals = []    #储存曲面的法向量
    
    # TODO：作业2
    for i in range(point_cloud_raw.shape[0]):
        # search_knn_vector_3d函数，取10个临近点进行曲线拟合
        # 输入值[每一点，x]，返回值 [int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        [_,idx,_] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i],10) 
        # asarray和array 一样 但是array会copy出一个副本，asarray不会，节省内存
        k_nearest_point = np.asarray(point_cloud_o3d.points)[idx, :]  # 找出每一点的10个临近点，类似于拟合成曲面，然后进行PCA找到特征向量最小的值，作为法向量
        w, v = PCA(k_nearest_point)
        normals.append(v[:, 2])

    normals = np.array(normals, dtype=np.float64)
    
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    # 法线向量显示：在显示窗口按n 可按 + - 更改点的大小（o3d）
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()