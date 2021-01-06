
import math
import numpy as np
# import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
## 计算“视线距离”到Z的修正
def dist_to_z(img_dist,cx,cy,f,cos_theta=None,ret_cos_theta=False):
    if cos_theta is None:
        hgt,wid=img_dist.shape
        
        x,y=np.meshgrid(range(wid),range(hgt))
        x=x.astype(np.float32)-cx
        y=y.astype(np.float32)-cy
        
        cos_theta=f/np.sqrt(x**2+y**2+f**2)
    img_z=img_dist*cos_theta
    
    return img_z if not ret_cos_theta else (img_z,cos_theta)

## 从z计算“视线距离”
def z_to_dist(img_z,cx,cy,f,cos_theta=None,ret_cos_theta=False):
    if cos_theta is None:
        hgt,wid=img_z.shape
        
        x,y=np.meshgrid(range(wid),range(hgt))
        x=x.astype(np.float32)-cx
        y=y.astype(np.float32)-cy
        
        cos_theta=f/np.sqrt(x**2+y**2+f**2)
    img_dist=img_z/cos_theta
    
    return img_dist if not ret_cos_theta else (img_dist,cos_theta)

## 计算点云
def img_z_to_pc(img_z,cx,cy,fx,fy,tab_x=None,tab_y=None,ret_tab=False):
    if tab_x is None or tab_y is None:
        hgt,wid=img_z.shape
        
        x,y=np.meshgrid(range(wid),range(hgt))
        x=x.astype(np.float32)-cx
        y=y.astype(np.float32)-cy
        
        tab_x=x/fx 
        tab_y=y/fy
    
    pc_x=img_z*tab_x    #   X=Z*(u-cx)/fx
    pc_y=img_z*tab_y    #   Y=Z*(v-cy)/fy
    pc_z=img_z
    pc=np.array([pc_x.ravel(),pc_y.ravel(),pc_z.ravel()]).T
    return pc if not ret_tab else (pc,tab_x,tab_y)

## 从深度图得到点云
def dist_to_pc(img_dist,cx,cy,fx,fy): 
    return img_z_to_pc(dist_to_z(img_dist,cx,cy,(fx+fy)/2.0),cx,cy,fx,fy)

## 将点云pc反向映射回到深度图。
#       u=X*fx/Z+cx
#       v=Y*fy/Z+cy
def pc_to_img_z(pc,fx,fy,cx,cy,hgt,wid,eps=1.0e-16,ret_valid=False):
    z=pc[:,2]
    z[np.abs(z)<eps]=eps    # 防止后面的除零错

    # 反向映射到像素坐标位置
    u=np.round(pc[:,0]*fx/z+cx).astype(int)
    v=np.round(pc[:,1]*fy/z+cy).astype(int)
    
    # 滤除超出图像尺寸的无效像素
    valid=np.bitwise_and(np.bitwise_and((u>=0),(u<wid)),
                         np.bitwise_and((v>=0),(v<hgt)))
    u_valid=u[valid]
    v_valid=v[valid]
    z_valid=z[valid]
    
    # 按距离填充生成深度图，近距离覆盖远距离
    img_z=np.full((hgt, wid),np.inf)        
    for ui,vi,zi in zip(u_valid,v_valid,z_valid):
        img_z[vi,ui]=min(img_z[vi,ui],zi)   # 近距离像素屏蔽远距离像素
    valid=np.bitwise_and(~np.isinf(img_z),img_z>0)
    return img_z if not ret_valid else (img_z,valid)

## 小洞和“透射”消除
def remove_hole(img_z):
    img_z_shift=np.array([img_z,\
                          np.roll(img_z, 1,axis=0),\
                          np.roll(img_z,-1,axis=0),\
                          np.roll(img_z, 1,axis=1),\
                          np.roll(img_z,-1,axis=1)])
    img_z_new=np.min(img_z_shift,axis=0)
    return img_z_new

## 深度图转成为RGB伪彩色
def dep_to_rgb(img,dmin=None,dmax=None,cmap=None,ret_cmap=False):
    if dmin is None: dmin=np.min(img)
    if dmax is None: dmax=np.max(img)
    if cmap is None:
        r=[255]*85+list(range(255,0,-3))+[0]*86
        g=list(range(0,255,3))+[255]*86+list(range(255,0,-3))
        b=[0]*85+list(range(0,255,3))+[255]*86
        cmap=np.array([r,g,b]).T
        cmap=np.flipud(cmap)

    hgt,wid=img.shape
    img_uint8=np.clip(np.round((img-dmin)/(dmax-dmin)*256),0,255).astype(np.uint8)
    img_rgb=np.zeros((hgt,wid,3),dtype=np.uint8)
    x,y=np.meshgrid(range(wid),range(hgt))
    for ix,iy in zip(x.ravel(),y.ravel()):
        img_rgb[iy,ix,:]=cmap[img_uint8[iy,ix]]
    
    return img_rgb if not ret_cmap else (img_rgb,cmap)
    
####################
# 单元测试
####################
def test():
    # 相机参数, 59.8度640x480相机
    CAM_FX,CAM_FY   = 795.209,793.957
    CAM_CX,CAM_CY   = 332.031,231.308
    CAM_DVEC = np.array([-0.33354, 0.00924849, -0.000457208, -0.00215353, 0.0])

    img_dep=np.genfromtxt('/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/02.3D传感器技术原理/CH2-HW/img_dep_640x480.csv',delimiter=',').astype(np.float32)
    IMG_HGT,IMG_WID = img_dep.shape
    if True:
        plt.imshow(img_dep,cmap='jet')
        plt.show()

    img_z=dist_to_z(img_dep,CAM_CX,CAM_CY,(CAM_FX+CAM_FY)/2.0)   # 计算“视线距离”到Z的修正

    # 计算点云
    pc=img_z_to_pc(img_z,CAM_CX,CAM_CY,CAM_FX,CAM_FY)

    # 显示点云
    if True:
        from pc_view import pc_view
        pc_view(pc,\
            CAM_FX,CAM_FY,CAM_CX,CAM_CY,IMG_WID,IMG_HGT,\
            cz=0.5, \
            dmin=0.5,dmax=1.5)

    # 旋转平移
    # from pc_trans import *
    pc_new=pc+[0,0,-0.5]
    R=calc_matrix_roty(math.radians(30))
    pc_new=np.dot(pc_new,R)
    pc_new=pc_new+[0,0,1]

    if False:
        np.savetxt('/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/03.3D数据表示和转换/CH3-HW/pc_rot.csv', pc_new, fmt='%.8f', delimiter=',', newline='\n', header='', footer='', comments='# ')

    # 显示平移旋转后的点云
    if True:
        from pc_view import pc_view
        pc_view(pc_new,\
            CAM_FX,CAM_FY,CAM_CX,CAM_CY,IMG_WID,IMG_HGT,\
            cz=0.5, \
            dmin=0.5,dmax=1.5)

    # 点云重新投影为深度图
    img_z_new=pc_to_img_z(pc_new,CAM_FX,CAM_FY,CAM_CX,CAM_CY,IMG_HGT,IMG_WID)
    plt.imshow(img_z_new,cmap='jet')
    plt.show()

    # “透射”消除
    img_z_new2=remove_hole(img_z_new)
    plt.imshow(img_z_new2,cmap='jet')
    plt.show()


def show_pc(path):
    pc = np.genfromtxt(path,delimiter=',').astype(np.float32)
    print(pc.shape)
    ax=plt.figure().gca(projection='3d')
    ax.plot(pc[:,0],pc[:,1],pc[:,2],'.',markersize=0.1)
    plt.show()
    
if __name__ == '__main__':
    show_pc("/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/06.特征提取、前后景分割、物体检测和识别/check/3.pc")
    






