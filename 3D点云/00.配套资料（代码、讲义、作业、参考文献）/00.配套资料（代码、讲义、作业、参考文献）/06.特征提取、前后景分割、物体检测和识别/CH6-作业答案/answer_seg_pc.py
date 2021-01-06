#!/usr/bin/python3
# coding=utf-8

import IPython

import sys,math,time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1234)

# 装载私有库
from pc_view   import pc_view
from pc_to_dep import *

# 相机参数
CAM_FX,CAM_FY=3.664861e+02, 3.664861e+02
CAM_CX,CAM_CY=2.543334e+02,1.967047e+02
CAM_HGT,CAM_WID=424,512
CAM_F=(CAM_FX+CAM_FY)*0.5

# 程序常量
TH_DIST     = 1.5      # 距离门限，截取这个距离内的点云
TH1         = 0.012    # 背景分离门限
TH2         = 0.008    # 离群点滤除参数和物体点云聚合的距离门限
TH_OBJ_SZ   = 500      # 尺寸门限，滤除点云太小的物体

print('loading CSV...')
pc   =np.genfromtxt('pc.csv'   ,delimiter=',').astype(np.float32)
pc_bg=np.genfromtxt('pc_bg.csv',delimiter=',').astype(np.float32)

pc_view(pc,CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT,cz=1,dmin=0,dmax=2)
pc_view(pc_bg,CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT,cz=1,dmin=0,dmax=2)

## 仅截取近距离物体
print('clipping by distance...')
pc_sel=np.array([p for p in pc if p[2]<TH_DIST])

## 背景提取
print('nn searching...')
flann = cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=50))
matches = flann.knnMatch(pc_sel,pc_bg,k=1)

## 保留非背景点
print('filtering...')
pc_sel=np.array([p for p,m in zip(pc_sel,matches) if m[0].distance>TH1])

## 滤除离群点
print('noise filter by distance...')
matches = flann.knnMatch(pc_sel,pc_sel,k=5)
pc_sel=np.array([p for p,m in zip(pc_sel,matches) if max([i.distance for i in m])<TH2])
pc_view(pc_sel,CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT,cz=1,dmin=0,dmax=2)

## 基于邻域拓展的物体分割
def pc_obj_merge(pc,p0,k,r,mask=None,ret_idx=False):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    
    # 指示尚未分割的点云点(这里用copy()是为了防止修改输入的mask对象)
    mask=np.ones(len(pc),dtype=bool) if mask is None else mask.copy()
            
    pc_chk=[p0]
    idx_out=[]
    while len(pc_chk)>0:
        p=pc_chk.pop()
        # 找到半径r领域内的未检查过的点的序号
        idx_nn=[m.trainIdx for m in flann.knnMatch(p,pc,k=k)[0] \
                           if m.distance<r and mask[m.trainIdx]]
        idx_out+=idx_nn                     # 新增的点云序号加入输出集合
        pc_chk +=[pc[i] for i in idx_nn]    # 新增点云加入待检测集合
        mask[idx_nn]=False                  # 标注已被处理的点
    return (pc[idx_out],idx_out) if ret_idx else pc[idx_out]

## 逐个提取里面的物体
pc_obj_list=[]
mask=np.ones(len(pc_sel),dtype=bool)   
while np.sum(mask)>0:
    # 找到种子点对应的物体
    p0=pc_sel[np.flatnonzero(mask)[0]]  # 选取种子点
    
    pc_obj,idx_obj=pc_obj_merge(pc_sel,p0,k=10,r=TH1,mask=mask,ret_idx=True)
    pc_obj_list+=[pc_obj]
    mask[idx_obj]=False
    print('p0:',p0,', size:',len(idx_obj))

# 过滤太小的物体
print('removing small objects...')
pc_obj_list_sel=[obj for obj in pc_obj_list if len(obj)>500]

# 显示
for n,pc_obj in enumerate(pc_obj_list_sel):
    np.savetxt('pc_obj%d.csv'%n,pc_obj,fmt='%.12f',delimiter=',',newline='\n')
    pc_view(pc_obj,CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT,cz=1,dmin=0,dmax=2)


# 映射回深度图，用不同颜色显示
img_rgb=np.zeros((CAM_HGT,CAM_WID,3),dtype=np.uint8)
for pc_obj in pc_obj_list_sel:
    img=pc_to_img_z(pc_obj,CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_HGT,CAM_WID)
    img_rgb[(~np.isinf(img)).tolist(),:] =np.random.randint(0,256,3)
plt.imshow(img_rgb)
plt.show()

## 加载刚才保存的物体点云
pc_obj0=np.genfromtxt('pc_obj0.csv'   ,delimiter=',').astype(np.float32)
pc_obj1=np.genfromtxt('pc_obj1.csv'   ,delimiter=',').astype(np.float32)
pc_obj2=np.genfromtxt('pc_obj2.csv'   ,delimiter=',').astype(np.float32)

pc_view(pc_obj0,CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT,cz=1,dmin=0,dmax=2)
pc_view(pc_obj1,CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT,cz=1,dmin=0,dmax=2)
pc_view(pc_obj2,CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT,cz=1,dmin=0,dmax=2)
