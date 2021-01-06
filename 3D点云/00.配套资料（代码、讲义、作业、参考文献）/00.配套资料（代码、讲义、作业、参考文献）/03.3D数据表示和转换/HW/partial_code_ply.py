# 使用简易三角剖分算法，从深度图生成ply数据文件
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

# 相机参数, 59.8度640x480相机
CAM_WID,CAM_HGT = 640,480           # 图像尺寸
CAM_FX,CAM_FY   = 795.209,793.957   # fx/fy
CAM_CX,CAM_CY   = 332.031,231.308   # cx/cy

def read_file(show=True):
    # 加载灰度图数据
    img_gray=np.genfromtxt('head_gray.csv', delimiter=',').astype(np.float32)
    # 加载深度图数据
    img_dep=np.genfromtxt('head_dep.csv', delimiter=',').astype(np.float32)
    
    if(show):
        # 显示灰度图和强度图
        plt.subplot(1,2,1)
        plt.imshow(img_gray,cmap='gray')
        plt.title('gray')
        plt.subplot(1,2,2)
        plt.imshow(img_dep,cmap='jet')
        plt.title('depth')
        plt.show()
    return img_dep, img_gray

def prepare(img_dep, img_gray):
    # 深度图转点云
    x,y=np.meshgrid(range(CAM_WID),range(CAM_HGT))
    x=x.astype(np.float32)-CAM_CX
    y=y.astype(np.float32)-CAM_CY
    img_z=img_dep.copy()
    
    if False:   # 如需矫正视线到Z的转换
        f=(CAM_FX+CAM_FY)/2.0
        img_z*=f/np.sqrt(x**2+y**2+f**2)
    pc_x=img_z*x/CAM_FX     #   X=Z*(u-cx)/fx
    pc_y=img_z*y/CAM_FY     #   Y=Z*(v-cy)/fy
    pc=np.array([pc_x.ravel(),pc_y.ravel(),img_z.ravel()]).T

    # 亮度图上限调整
    img_gray=(np.round(img_gray/np.max(img_gray)*255)).astype(int)
    return pc, img_gray

def write_plyfile(pc, img_gray):
    # 生成文件数据
    idx=np.arange(CAM_HGT*CAM_WID).reshape(CAM_HGT,CAM_WID)
    tri_up=np.array([idx,
                     np.roll(idx,-1,axis=0),
                     np.roll(np.roll(idx,-1,axis=0), 1,axis=1)])
    tri_dn=np.array([idx,
                     np.roll(idx, 1,axis=1),
                     np.roll(np.roll(idx, 1,axis=1),-1,axis=0)])
    
    print(tri_up[:,0,0])
    fp=open('head.ply','wt')
    fp.write('ply\nformat ascii 1.0\nelement vertex %d\n'%(CAM_HGT*CAM_WID))
    fp.write('property float32 x\nproperty float32 y\nproperty float32 z\nproperty uint8 red\nproperty uint8 green\nproperty uint8 blue\n')
    fp.write('element face %d\n'%((CAM_HGT-1)*(CAM_WID-1)*2))
    fp.write('property list uint8 int32 vertex_index\nend_header\n')

    for p,c in zip(pc,img_gray.ravel()):
        fp.write('%f %f %f %d %d %d\n'%(p[0],p[1],p[2],c,c,c))

    for y in range(0,CAM_HGT-1):
        for x in range(1,CAM_WID):
            fp.write('3 %d %d %d\n'%(tri_up[0,y,x],tri_up[1,y,x],tri_up[2,y,x]))
            fp.write('3 %d %d %d\n'%(tri_dn[0,y,x],tri_dn[1,y,x],tri_dn[2,y,x]))
    fp.close()

def show_plyfile():
    pc = o3d.io.read_point_cloud("/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/03.3D数据表示和转换/CH3-HW/head.ply")
    o3d.visualization.draw_geometries([pc])
    
if __name__ == '__main__':
    img_dep, img_gray = read_file(False)
    pc, img_gray = prepare(img_dep, img_gray)
    write_plyfile(pc, img_gray)
    #show_plyfile()