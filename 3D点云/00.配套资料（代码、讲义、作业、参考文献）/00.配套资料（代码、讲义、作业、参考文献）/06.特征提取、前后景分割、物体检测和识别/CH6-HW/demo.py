import cv2 
import math
import numpy as np
# import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D

# 边沿检测——基于深度图的边沿检测
# 1. 待边沿检测的平面法向量计算
# 2. 点云旋转至法向量垂直屏幕
# 3. 点云重投影到深度图
# 4. 重投影深度图上的边沿检测
# 5. 重投影深度图上对应边界像素转成点云
def img_z_edge_det(img_z, win, th1=100, th2=200,mode='sobel'):
    if mode == 'sobel':
        sobelx = cv2.Sobel(img_z.astype(np.float32), cv2.CV_32F, 1, 0, ksize=win)
        sobely = cv2.Sobel(img_z.astype(np.float32), cv2.CV_32F, 0, 1, ksize=win)
        return np.sqrt(sobelx**2 + sobely**2)

    if mode == 'canny':
        vmax = np.max(img_z)
        vmin = np.min(img_z)
        img_u8 = ((img_z-vmin)/(vmax-vmin)*255).astype(np.uint8)
        return cv2.Canny(img_u8, th1, th2)
    if mode == 'var':
        return c2.blur(img_z**2, (win, win))-cv2.blur(img_z, (win,win))**2

# 区域生长法表面点云提取
def pc_obj_merge():
    pass