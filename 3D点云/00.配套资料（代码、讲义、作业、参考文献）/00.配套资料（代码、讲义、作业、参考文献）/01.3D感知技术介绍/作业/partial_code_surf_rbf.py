#!/usr/bin/python3
# coding=utf-8

import numpy as np

np.random.seed(4321)

####################
# API
####################

## 功能描述：
#   逐点计算表面法向量，对每个点用他的k近邻拟合平面计算法向量
# 输入:
#   pc  点计算法向量点云，每行对应一个点的x/y/z坐标
#   k   用于拟合平面计算法向量的近邻数量
#   func_weight 权重函数，用于拟合平面时，对每个近邻点的重要性评估
# 输出：
#   pcn 其中每一行对应pc的点的法向量估计
def find_surf_dir_with_knn(pc,k,func_phi=None):
    pcn=[]
    for p in pc:    # 逐行取出pc内容，p对应pc中每个点的坐标
        idx,dist=find_knn(p,pc,k)                       # 找到p的k近邻
        w=None if func_phi is None else func_phi(dist)  # 按距离加权
        pcn.append(find_surf_dir(pc[idx,:],w))          # 计算法向量并记录
    return np.array(pcn).reshape(-1,3)


## 功能描述：
#   根据点云对应的平面找到法向量
#   用特征值分解计算平面方向
# 输入：
#   pc  点计算法向量点云，每行对应一个点的x/y/z坐标
#   w   pc中每个点的重要性权重
def find_surf_dir(pc,w=None):
    pc0=pc-np.mean(pc,axis=0)
    if w is not None: pc0*=w.reshape(-1,1)
    E,F=np.linalg.eig(np.dot(pc0.T,pc0))# E:特征值, F:特征向量
    return F[:,np.argmin(E)].ravel()

        
## 功能描述：
#   简易的最近邻搜索，从点云pc中寻找点p的k近邻
# 输入：
#   p   待近邻查询的点的坐标
#   pc  点计算法向量点云，每行对应一个点的x/y/z坐标
#   k   查询的近邻数量
# 输出：
#   返回点k近邻的点云序号idx和对应的距离dist
def find_knn(p,pc,k=1):
    dist=np.linalg.norm(pc-p.reshape(1,3),axis=1)
    idx=np.argsort(dist)[:k]
    return idx,dist[idx]


## 点云按法向量扩展
def pc_ext(pc,p0,d,k,func_phi=None):
    pcn=find_surf_dir_with_knn(pc,k,func_phi)
    
    # 法向量方向修正，这里简化了问题，仅仅要求法向量背离给定的中心点p0
    pcn=np.array([n*np.sign(np.dot(n,p-p0)) \
                    for p,n in zip(pc,pcn)])
    pc_out=pc+pcn*d   # 内点云
    pc_in =pc-pcn*d   # 外点云
                
    return pc_in,pc_out

## 根据拟合模型计算SDF
def calc_SDF(p,pc,w,func_phi): return np.sum(func_phi(np.linalg.norm(pc-p,axis=1))*w)

## 构建SDF
#   pc  点云
#   p0  点云内点中心（用于确定法向量方向）
#   d   点云沿法向量正负方向拓展的距离
#   k   计算法向量的knn数量
def construct_SDF(pc,p0,d,k,func_phi=None,verbose=False):
    # 点云按法向量方向拓展
    pc_in,pc_out=pc_ext(pc,p0=P0,d=D,k=K,func_phi=func_phi)   
    pc_all=np.vstack((pc,pc_out,pc_in))
    if verbose: 
        plot_pc(pc_in ,markersize=2,color='g',show=False)
        plot_pc(pc_out,markersize=2,color='r',show=False)
        plot_pc(pc    ,markersize=2,color='k',show=False)
        plt.show()
        
    num=pc.shape[0]
    # 解线性方程计算SDF
    M=np.array([func_phi(np.linalg.norm(pc_all[r]-pc_all,axis=1)) \
                for r in range(3*num)])
    v=np.array([   np.zeros(num),\
                 D*np.ones (num),\
                -D*np.ones (num)]).ravel()
    if verbose: print('[INF] solve linear equ....')
    if verbose: print('[INF] cond(M):',np.linalg.cond(M))
    invM=np.linalg.pinv(M)
    w=np.dot(invM,v).ravel()
    return pc_all,w


## 距离函数,根据距离r计算并返回权重
def func_phi_impl(r): 
    w=1.0/(0.1+r**GAMMA)
    return w/np.sum(w)
    
    
####################
# SDF构建入口
####################

############################
# 下面的3个数据需要你尝试，
# 找到合适的数值
K=      # 计算平面法向量需要的近邻数
D=      # 沿着法向量方向拓展点云的位移量
GAMMA=  # 和距离函数有关的参数
####################

P0=(0,0,0)  # 内点中心，用于确定法向量方向

# 加载并显示立方体表面点云
print('loading pc...')
pc=np.genfromtxt('pc_cube.csv',delimiter=',').astype(np.float32)
        
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ax = plt.figure().gca(projection='3d')
ax.plot(pc[:,0],pc[:,1],pc[:,2],'r.',markersize=1)
plt.show()

# 构建SDF函数
pc_all,w=construct_SDF(pc,p0=P0,d=D,k=K,func_phi=func_phi_impl,verbose=False)

# 计算SDF函数在给定空间坐标coord的值，并显示
# 这里的空间坐标coord是在z=0的平面上的矩阵格点，共100x100个点
coord=np.array([(x,y,0) for x in np.linspace(-1.5,1.5,100) for y in np.linspace(-1.5,1.5,100)])

img_cut=np.array([calc_SDF((x,y,z),pc_all,w,func_phi=func_phi_impl) for x,y,z in coord])
img_cut.shape=100,100

# 显示并保存计算得到的SDF函数值
plt.clf()
plt.imshow(img_cut,cmap='jet')
plt.colorbar()
plt.title('SDF')
plt.savefig('sdf.png')  # 保存图片
plt.show()

