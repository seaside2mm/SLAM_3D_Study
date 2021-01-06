# https://blog.csdn.net/weixin_41281151/article/details/107125844

# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from pandas import DataFrame

def Point_Cloud_Show(points):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], cmap='spectral', s=2, linewidths=0, alpha=1, marker=".")
    plt.title('Point Cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def Point_Show(pca_point_cloud):
    x = []
    y = []
    pca_point_cloud = np.asarray(pca_point_cloud)
    for i in range(10000):
        x.append(pca_point_cloud[i][0])
        y.append(pca_point_cloud[i][1])
    plt.scatter(x, y)
    plt.show()



# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size,filter_mode):
    filtered_points = []
    # TODO: 作业3
    #step1 计算边界点
    x_max, y_max, z_max = np.amax(point_cloud,axis=0)      #计算 x,y,z三个维度的最值
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    #step2 确定体素的尺寸
    size_r = leaf_size
    #step3 计算每个 volex的维度
    Dx = (x_max - x_min)/size_r
    Dy = (y_max - y_min)/size_r
    Dz = (z_max - z_min)/size_r
    #step4 计算每个点在volex grid内每一个维度的值
    h = list()
    for i in range(len(point_cloud)):
        hx = np.floor((point_cloud[i][0] - x_min)/size_r)
        hy = np.floor((point_cloud[i][1] - y_min)/size_r)
        hz = np.floor((point_cloud[i][2] - z_min)/size_r)
        h.append(hx + hy*Dx + hz*Dx*Dy)
    #step5 对h值进行排序
    h = np.array(h)
    h_indice  = np.argsort(h)   #提取索引
    h_sorted = h[h_indice]      #升序
    count = 0 #用于维度的累计
    #将h值相同的点放入到同一个grid中，并进行筛选
    for i  in range(len(h_sorted)-1):      #0-19999个数据点
        if h_sorted[i] == h_sorted[i+1]:   #当前的点与后面的相同，放在同一个volex grid中
            continue
        else:
            if(filter_mode == "centroid"):    #均值滤波
                point_idx = h_indice[count: i+1]
                filtered_points.append(np.mean(point_cloud[point_idx],axis=0))   #取同一个grid的均值
                count = i
            elif(filter_mode == "random"):  #随机滤波
                point_idx = h_indice[count: i+1]
                random_points =  random.choice(point_cloud[point_idx])
                filtered_points.append(random_points)
                count = i
                
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    ############################## 读取数据 ##############################
    point_cloud_raw = np.genfromtxt(r"/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/03.3D数据表示和转换/CH3-HW/airplane_0001.txt", delimiter=",")
    point_cloud_raw = DataFrame(point_cloud_raw[:,0:3])   # 为 xyz的 N*3矩阵
    point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud  = np.array(point_cloud_raw)
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    #o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    ############################## 读取数据 ##############################
    points = np.array(point_cloud_o3d.points)
    point_cloud_o3d_filter = o3d.geometry.PointCloud() 
    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(points, 0.05, "centroid")   #centroid or random
    point_cloud_o3d_filter.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波前后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])
    o3d.visualization.draw_geometries([point_cloud_o3d_filter])
if __name__ == '__main__':
    main()