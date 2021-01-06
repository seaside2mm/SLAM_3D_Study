
import numpy as np
from numpy import linalg as np_lin

####################
# 最近邻查找，
# 使用简单但低效的遍历法
####################

## 简易的最近邻搜索，从点云pc中寻找点p的k近邻
# 返回点k近邻的点云序号和计算得到的距离
def find_knn(p,pc,k=1):
    pc_diff=pc-p.reshape(1,3)
    dist=np_lin.norm(pc_diff,axis=1)
    idx=np.argsort(dist)[:k]
    return idx,dist[idx]

## 简易的最近邻搜索，从点云pc中寻找点p的距离小于等于r的近邻
# 返回点近邻的点云序号和计算得到的距离
def find_nn_by_r(p,pc,k=1):
    pc_diff=pc-p.reshape(1,3)
    dist=np_lin.norm(pc_diff,axis=1)
    
    sel=dist<=r
    idx=np.arange(len(pc))
    return idx[sel],dist[sel]
