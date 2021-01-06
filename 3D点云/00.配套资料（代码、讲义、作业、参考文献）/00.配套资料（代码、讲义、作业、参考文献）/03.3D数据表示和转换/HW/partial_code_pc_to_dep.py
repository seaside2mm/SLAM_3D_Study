# 点云重投影到深度图
# 
# u = X*f_x/Z+c_x
# Done by seaside in 10.11
# Takes about 1.5h

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 相机参数, 59.8度640x480相机
CAM_WID,CAM_HGT = 640,480           # 重投影到的深度图尺寸
CAM_FX,CAM_FY   = 795.209,793.957   # fx/fy
CAM_CX,CAM_CY   = 332.031,231.308   # cx/cy

def read_file(show=True):
    # 加载点云数据
    pc=np.genfromtxt('pc_rot.csv', delimiter=',').astype(np.float32)
    if(show):
        # 显示加载的点云
        ax = plt.figure(1).gca(projection='3d')
        ax.plot(pc[:,0],pc[:,1],pc[:,2],'b.',markersize=0.1)
        plt.title('point cloud')
        plt.show()  
    return pc

def pc_to_depth(pc, show=True, optimize=True):
    # 计算视线距离到z的修正
    # f = (CAM_FX+CAM_FY)/2.0
    # x = np.tile(np.arange(CAM_WID)-CAM_CX,CAM_HGT).reshape(CAM_HGT, CAM_WID)
    # y = np.repeat(np.arange(CAM_HGT)-CAM_CY,CAM_WID).reshape(CAM_HGT, CAM_WID)
    # cos_theta = f / np.sqrt(x ** 2 + y ** 2 + f ** 2)
    # img_z = img_depth *cos_theta

    dep_rot = np.zeros((CAM_HGT, CAM_WID)) 
    
    #利用转换公式  
    u = np.floor(np.array(pc[:,0]*CAM_FX/pc[:,2]+CAM_CX)).T
    v = np.floor(np.array(pc[:,1]*CAM_FY/pc[:,2]+CAM_CY)).T
    
    for i in range(len(u)):
        if(int(u[i] >= CAM_WID)): u[i] = CAM_WID - 1
        if(int(v[i] >= CAM_HGT)): v[i] = CAM_HGT - 1
        if(dep_rot[int(v[i]), int(u[i])] != 0 and dep_rot[int(v[i]), int(u[i])] < pc[i,2]): 
            continue
        else: dep_rot[int(v[i]), int(u[i])] = pc[i,2]
    
    if(optimize):
        kernel = np.ones((3,3))
        dep_rot = cv2.dilate(dep_rot, kernel) # 透射”问题的解决——使用灰度图形态学滤波
        dep_rot[dep_rot<0.5]=math.inf  # 解决无效数据
    
    if(show):
        # 随机生成的dep_rot, 用于演示数据保存成csv以及加载csv文件以及显示
        # np.random.seed(1234)
        # dep_rot=cv2.blur(np.random.rand(CAM_HGT,CAM_WID),(50,50))

        # 保存重新投影生成的深度图dep_rot
        np.savetxt('dep_rot.csv',dep_rot,fmt='%.12f',delimiter=',',newline='\n')
        # 加载刚保存的深度图dep_rot并显示
        img=np.genfromtxt('dep_rot.csv', delimiter=',').astype(np.float32)
        plt.imshow(img,cmap='jet')
        plt.show()
    
def answer():
    EPS=1.0e-16
    # 加载点云数据
    pc=np.genfromtxt('pc_rot.csv', delimiter=',').astype(np.float32)

    # 滤除镜头后方的点
    valid=pc[:,2]>EPS
    z=pc[valid,2]
            
    # 点云反向映射到像素坐标位置
    u=np.round(pc[valid,0]*CAM_FX/z+CAM_CX).astype(int)
    v=np.round(pc[valid,1]*CAM_FY/z+CAM_CY).astype(int)
        
    # 滤除超出图像尺寸的无效像素
    valid=np.bitwise_and(np.bitwise_and((u>=0),(u<CAM_WID)),  # u满足边界，v满足边界，选取uv都满足的坐标
                        np.bitwise_and((v>=0),(v<CAM_HGT)))
    u,v,z=u[valid],v[valid],z[valid]

    # 按距离填充生成深度图，近距离覆盖远距离
    img_z=np.full((CAM_HGT, CAM_WID),np.inf)        
    for ui,vi,zi in zip(u,v,z):
        img_z[vi,ui]=min(img_z[vi,ui],zi)   # 近距离像素屏蔽远距离像素

    # 小洞和“透射”消除
    # 透射:灰度图形态学滤波腐蚀运算类似卷积滤波，但保留滑动窗口内灰度最低值,min 𝑓(𝑥 + 𝑠, 𝑦 + 𝑡) 
    # 这里的深度图取代灰度图，使用腐蚀运算当窗口中心点距离点比周围远时，用周围的替代窗口中心点
    img_z_shift=np.array([img_z,\
                        np.roll(img_z, 1,axis=0),\  
                        np.roll(img_z,-1,axis=0),\
                        np.roll(img_z, 1,axis=1),\
                        np.roll(img_z,-1,axis=1)])
    img_z=np.min(img_z_shift,axis=0)  # (5, 480, 640)
    plt.imshow(img_z,cmap='jet')
    plt.show()

if __name__ == '__main__':
    # pc = read_file(False)
    # pc_to_depth(pc, True)
    
    answer()

