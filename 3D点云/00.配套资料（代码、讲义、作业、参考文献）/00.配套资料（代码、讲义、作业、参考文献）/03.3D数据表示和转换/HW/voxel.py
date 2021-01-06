# 3D数据的体像素存储
# Done by Seaside in 10.11
# Take about 1h 

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pc_to_voxel(pc,                       # 点云数组（numpy数组）， 每一行对应一个点       
                pstart = (0.0, 0.0, 0.0), # 体素对应立方体顶点最小值 
                pend   = (1.0, 1.0, 1.0), # 体素对应立方体顶点最大值
                grid_size = 0.5e-2,       # 空间分割的小立方体边长
                verbose = False): 
             
    # 体像素划分的空间的三个坐标轴对应的小立方体数量
    num_x = int(math.ceil((pend[0] - pstart[0]) / grid_size))
    num_y = int(math.ceil((pend[1] - pstart[1]) / grid_size))
    num_z = int(math.ceil((pend[2] - pstart[2]) / grid_size))

    # 点云坐标转化成体像素数组元素下标
    if verbose: print('[INF] Quantization')
    pc0 = pc - np.array (pstart) # 得到点云相对于pstart位置的坐标
    pc_q = np.round(pc0 / grid_size).astype(int) # 点云坐标以立方体边长为单位量化，得到数值下标
    
    # 滤除体像素空间外的像素
    if verbose: print('[INF] Trimming')
    valid = (pc_q[:,0]>0)*(pc_q[:,0]<num_x)*\
            (pc_q[:,1]>0)*(pc_q[:,2]<num_z)*\
            (pc_q[:,2]>0)*(pc_q[:,1]<num_y)
    pc_q = pc_q[valid, :]
    
    # 构建三维数组存放体像素
    voxel_mat = np.zeros((num_x, num_y, num_z), dtype=bool)
    if verbose: print('[INF] Contructing 3D array')
    for x,y,z in pc_q: voxel_mat[x,y,z] = True #设置体像素值
    
    return voxel_mat


def voxel_to_pc(voxels):
    [x,y,z] = voxels.shape
    pointcloud = voxels.nonzero()
    return pointcloud

def test():
    # 生成测试数据，球壳
    N = 5000
    pc = np.random.rand(N,3)*2.0-1.0
    pc = pc/np.linalg.norm(pc,axis=1).reshape(N,1)*0.5
    
    #绘制生成的点云
    ax = plt.figure(1).gca(projection='3d')
    ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], 'b.', markersize=0.5)
    plt.show()

    #构建体像素矩阵
    voxels = pc_to_voxel(pc, (-1, -1, -1), (1, 1, 1), 0.1, True)
    
    #绘制
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors='b', edgecolor='k')
    plt.show()

    pc2 = voxel_to_pc(voxels)
    #绘制生成的点云
    ax = plt.figure(1).gca(projection='3d')
    ax.plot(pc2[0], pc2[1], pc2[2], 'b.', markersize=0.5)
    plt.show()
    
if __name__ == "__main__":
    test()
    