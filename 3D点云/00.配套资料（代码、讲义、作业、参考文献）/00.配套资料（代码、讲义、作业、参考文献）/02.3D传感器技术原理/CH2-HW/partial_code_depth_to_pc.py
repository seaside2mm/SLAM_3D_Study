# 深度图转换为点云
# FINISHED IN 2020.10.11 BY SEASIDE
# TAKES ABOUT TWO HOURS

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## 生成点云使用的相机参数如下：
CAM_WID,CAM_HGT = 640,480           # 深度图img的图像尺寸
CAM_FX,CAM_FY   = 795.209,793.957   # 相机的fx/fy参数
CAM_CX,CAM_CY   = 332.031,231.308   # 相机的cx/cy参数
CAM_DVEC = np.array([-0.33354, 0.00924849, -0.000457208, -0.00215353, 0.0]) # 相机镜头的矫正参数，用于cv2.undistort()的输入之一

def read_file(path, show=True):
    ## 从CSV文件加载深度图数据img并显示
    img=np.genfromtxt(path, delimiter=',').astype(np.float32)
    if(show):
        plt.imshow(img,cmap='jet')    # 显示加载的深度图
        plt.title('depth image')
        plt.show()
    return img

def computeXY(u, v, Z):
    ## math: X = Z(u-c_x)/f_x
    X = Z * (u -  CAM_CX) / CAM_FX
    Y = Z * (v -  CAM_CY) / CAM_FY
    return [X, Y, Z] 

def depth2pc(img):
    ## 在下面补充你的代码，从深度图img生成点云数据pc，并保存为CSV文件
    [h, w] = img.shape
    
    # simple way
    # pc = np.zeros((w*h, 3))
    # index = 0
    # for i in range(h):
    #     for j in range(w):
    #         pc[index] = np.array(computeXY(i, j, img[i,j]))
    #         index += 1
   
    # faster way
    u = np.tile(np.array(range(h)),(w,1)).T
    v = np.tile(np.array(range(w)),(h,1))
    X = img * (u -  CAM_CX) / CAM_FX
    Y = img * (v -  CAM_CY) / CAM_FY
    pc = np.array([X.ravel(), Y.ravel(), img.ravel()]).T
    return pc
    

def save_pointcloud(pc, show = True):
    ## 下面是保存CSV代码的例子以及显示点云的例子
    np.savetxt('pc.csv', pc, fmt='%.18e', delimiter=',', newline='\n')
    if(show):
        ## 从CSV文件加载点云并显示
        pc=np.genfromtxt('pc.csv', delimiter=',').astype(np.float32)
        ax = plt.figure(1).gca(projection='3d')
        ax.plot(pc[:,0],pc[:,1],pc[:,2],'b.',markersize=0.5)
        plt.title('point cloud')
        plt.show()   
        
if __name__ == '__main__':
    
    depth_img = read_file('match.csv', False)
    pc = depth2pc(depth_img)
    save_pointcloud(pc)

 

