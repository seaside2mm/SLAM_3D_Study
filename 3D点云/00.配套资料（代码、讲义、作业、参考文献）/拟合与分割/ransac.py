# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import open3d as o3d
import struct
import matplotlib.pyplot as plt
from pandas import DataFrame
import math
import random
from collections import defaultdict


# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    #初始化数据
    idx_segmented = []
    segmented_cloud = []
    iters = 100   #最大迭代次数  000002.bin：10
    sigma = 0.4     #数据和模型之间可接受的最大差值   000002.bin：0.5   000001.bin: 0.2  000000.bin: 0.15  002979.bin：0.15  004443.bin：0.4
    ##最好模型的参数估计和内点数目,平面表达方程为   aX + bY + cZ +D= 0
    best_a = 0
    best_b = 0
    best_c = 0
    best_d = 0
    pretotal = 0 #上一次inline的点数
    #希望的到正确模型的概率
    P = 0.99
    n = len(data)    #点的数目
    outline_ratio = 0.6   #e :outline_ratio   000002.bin：0.6    000001.bin: 0.5  000000.bin: 0.6   002979.bin：0.6
    for i in range(iters):
        ground_cloud = []
        idx_ground = []
        #step1 选择可以估计出模型的最小数据集，对于平面拟合来说，就是三个点
        sample_index = random.sample(range(n),3)    #重数据集中随机选取3个点
        point1 = data[sample_index[0]]
        point2 = data[sample_index[1]]
        point3 = data[sample_index[2]]
        #step2 求解模型
        ##先求解法向量
        point1_2 = (point1-point2)      #向量 poin1 -> point2
        point1_3 = (point1-point3)      #向量 poin1 -> point3
        N = np.cross(point1_3,point1_2)            #向量叉乘求解 平面法向量
        ##slove model 求解模型的a,b,c,d
        a = N[0]
        b = N[1]
        c = N[2]
        d = -N.dot(point1)
        #step3 将所有数据带入模型，计算出“内点”的数目；(累加在一定误差范围内的适合当前迭代推出模型的数据)
        total_inlier = 0
        pointn_1 = (data - point1)    #sample（三点）外的点 与 sample内的三点其中一点 所构成的向量
        distance = abs(pointn_1.dot(N))/ np.linalg.norm(N)     #求距离
        ##使用距离判断inline
        idx_ground = (distance <= sigma)
        total_inlier = np.sum(idx_ground == True)    #统计inline得点数
        ##判断当前的模型是否比之前估算的模型
        if total_inlier > pretotal:                                           #     log(1 - p)
            iters = math.log(1 - P) / math.log(1 - pow(total_inlier / n, 3))  #N = ------------
            pretotal = total_inlier                                               #log(1-[(1-e)**s])
            #获取最好得 abcd 模型参数
            best_a = a
            best_b = b
            best_c = c
            best_d = d

        # 判断是否当前模型已经符合超过 inline_ratio
        if total_inlier > n*(1-outline_ratio):
            break
    print("iters = %f" %iters)
    #提取分割后得点
    idx_segmented = np.logical_not(idx_ground)
    ground_cloud = data[idx_ground]
    segmented_cloud = data[idx_segmented]
    return ground_cloud,segmented_cloud

    # 屏蔽结束

    # print('origin data points num:', data.shape[0])
    # print('segmented data points num:', segmengted_cloud.shape[0])
    # return segmengted_cloud

def main():
    iteration_num = 1    #文件数

    # for i in range(iteration_num):
    filename = '/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/拟合与分割/1.bin'         #数据集路径
    print('clustering pointcloud file:', filename)

    origin_points = read_velodyne_bin(filename)   #读取数据点
    origin_points_df = DataFrame(origin_points,columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(origin_points)
    
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 地面分割
    ground_points, segmented_points = ground_segmentation(data=origin_points)

    ground_points_df = DataFrame(ground_points, columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_o3d_ground = o3d.geometry.PointCloud()
    point_cloud_o3d_ground.points = o3d.utility.Vector3dVector(ground_points)
    point_cloud_o3d_ground.paint_uniform_color([0, 0, 255])


    #显示segmentd_points示地面点云
    segmented_points_df = DataFrame(segmented_points, columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_o3d_segmented = o3d.geometry.PointCloud()
    point_cloud_o3d_segmented.points = o3d.utility.Vector3dVector(segmented_points)
    point_cloud_o3d_segmented.paint_uniform_color([255, 0, 0])

    o3d.visualization.draw_geometries([point_cloud_o3d_ground,point_cloud_o3d_segmented])

if __name__ == '__main__':
    main()