#!/usr/bin/python3
# coding=utf-8

import numpy as np

GRID_SIZE=0.01                                      # 格点尺寸

# 加载原始点云数据
pc=np.genfromtxt('pc.csv', delimiter=',').astype(np.float32)

###############################
# 需要你在下面填写代码实现点云降采样
# 下面的代码只是一个随机生成的点云例子，需要你修改
pc_new=np.random.rand(50000,3)*0.1-0.05+[0,0,1]
###############################

# 保存结果
np.savetxt('pc_new.csv', pc_new, fmt='%.8f', delimiter=',', newline='\n')


# 3D显示结果
pc_new=np.genfromtxt('pc_new.csv',delimiter=',').astype(np.float32) # 加载保存的数据

# 显示参数
CAM_WID,CAM_HGT = 640,480
CAM_FX,CAM_FY   = 795.209,793.957
CAM_CX,CAM_CY   = 332.031,231.308

from pc_view import pc_view
pc_view(pc, CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT)
pc_view(pc_new, CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT)

