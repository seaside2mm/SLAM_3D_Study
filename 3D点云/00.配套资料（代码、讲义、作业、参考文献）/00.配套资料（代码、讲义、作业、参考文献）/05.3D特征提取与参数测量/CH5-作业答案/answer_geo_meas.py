#!/usr/bin/python3
# coding=utf-8

import numpy as np

####################
# 几何参数提取
####################

## 功能描述：
#   计算直线模型参数：p=dt+m
# 返回
#   d：  方向
#   m：  直线上一点
def line_det_pc(pc):
    m=np.mean(pc,axis=0)    # 直线上一点（点云中心）
    pc0=pc-m                # 去均值
    C=np.dot(pc0.T,pc0)     # 协方差阵
    E,V=np.linalg.eig(C)    # E: 特征值, V: 特征向量
    idx=np.argmax(E)        # 排序
    d=V[:,idx].ravel()      # 最大特征值(对应直线方向)
    return d,m


## 功能描述：
#   计算平面模型参数：<n,p>=D
# 返回
#   n：  平面法向量方向
#   D：  原点到平面的距离（沿着法向量方向，可能是负值）
def plane_det_pc(pc):
    m=np.mean(pc,axis=0)    # 平面上一点（点云中心）
    pc0=pc-m                # 去均值
    C=np.dot(pc0.T,pc0)     # 协方差阵
    E,V=np.linalg.eig(C)    # E: 特征值, V: 特征向量
    idx=np.argmin(E)        # 排序
    n=V[:,idx].ravel()      # 最小特征值(对应平面法线方向，归一化过了)
    D=np.sum(n*m)           # 平面上的点m在法向量上投影长度
    return n,D


## 功能描述：
#   最小二乘法从球面点云获取球体的参数
# (x-cx)**2+(y-cy)**2+(z-cz)**2-R**2=0
# 输入：
#   pc:     待计算点坐标序列(每行对应一个点的X/Y/Z坐标)
# 输出：
#   球心坐标和半径
def sphere_det_pc(pc):
    V=np.hstack((2*pc,
                 np.ones((pc.shape[0],1))))
    b=np.sum(pc**2,axis=1).reshape(-1,1)
    # w=inv(V'V)*V'b
    w=np.dot(np.linalg.pinv(np.dot(V.T,V)),np.dot(V.T,b)).ravel()
    r=(w[3]+np.sum(w[:3]**2))**0.5
    return w[:3],r


## 功能描述：
#   检测空间圆环
# 输入：
#   pc:     待计算点坐标序列(每行对应一个点的X/Y/Z坐标)
# 输出：
#   p：环心坐标
#   r：圆环半径和半径
#   n：圆环法向量方向
def circle_det(pc):
    n,D=plane_det_pc(pc)        # n:平面法向量, D平面到原点距离
    c,_=sphere_det_pc(pc)       # c:球心
    
    p=pnt_to_plane_proj(c,n,D)  # 球心在平面上的投影是环心
    r=np.mean(np.linalg.norm(pc-p,axis=1))  # 半径
    return p,r,n

    
## 功能描述：
#   计算点v在直线上的投影坐标
# 直线参数方程：p=dt+m
def pnt_to_line_proj(v,d,m=0):
    return m+np.sum((v-m)*d)*d/np.sum(d**2)


## 功能描述：
#   计算点v在平面上的投影坐标
# 平面方程是<p,n>=D
def pnt_to_plane_proj(v,n,D):
    n0=n/np.linalg.norm(n)  # 长度归一化
    return v-n0*(np.sum(n*v)-D)


## 功能描述：
#   计算3D空间点到直线的距离
# 输入：
#   pc:     待计算点坐标序列（向量）
#   d,m:    直线参数模型: p=dt+m
# 输出：
#   每个点对应的距离
def pc_to_line_dist(d,m,pc):    
    return np.linalg.norm(np.cross(pc-m,pc-m-d),axis=1)/np.linalg.norm(d)


## 功能描述：
# 输入:
#   pc:     待计算距离的点云集合
#   n,D:    平面模型参数<n,p>=D
# 输出：
#   每个点对应的距离
def pc_to_plane_dist(n,D,pc):
    return np.abs(np.sum(pc*n,axis=1)-D)

# pc:   点云
# r:    点云子集尺寸（相对总的点云尺寸的比例系数）
# k:    近邻数量门限（相对总的点云尺寸的比例系数）
# th:   近邻距离门限
def line_det_pc_ransac(pc,r=0.2,k=0.3,th=0.1,it=20, it1=3, verbose=False):
    N=pc.shape[0]
    M,K=int(N*r),int(N*k)
    
    idx=np.arange(N)
    while it>0:
        if verbose: print('iteration',it)
        it-=1
        np.random.shuffle(idx)
        pc_sub=pc[idx[:M]]
        d,m=line_det_pc(pc_sub)
        dist=pc_to_line_dist(d,m,pc)
        if np.sum(dist<th)>K: break

    if it<=0: 
        print('RANSAC fail')
        return (None,None)
    
    if verbose: print('matched')
    while it1>0:
        it1-=1
        pc_sub=pc[dist<th]
        d,m=line_det_pc(pc_sub)
        dist=pc_to_line_dist(d,m,pc)
        
    return d,m

def plane_det_pc_ransac(pc,r=0.2,k=0.4,th=0.06,it=20, it1=3, verbose=False):
    N=pc.shape[0]
    M,K=int(N*r),int(N*k)
    
    idx=np.arange(N)
    while it>0:
        if verbose: print('iteration',it)
        it-=1
        np.random.shuffle(idx)
        pc_sub=pc[idx[:M]]
        n,D=plane_det_pc(pc_sub)
        dist=pc_to_plane_dist(n,D,pc)
        if np.sum(dist<th)>K: break

    if it<=0: 
        print('RANSAC fail')
        return (None,None)
    
    if verbose: print('matched')
    while it1>0:
        it1-=1
        pc_sub=pc[dist<th]
        n,D=plane_det_pc(pc_sub)
        dist=pc_to_plane_dist(n,D,pc)
        
    return n,D


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    
    # 加载原始点云数据
    pc=np.genfromtxt('pc_obj.csv', delimiter=',').astype(np.float32)

    # 显示加载的点云
    ax=plt.figure().gca(projection='3d')
    ax.plot(pc[:,0],pc[:,1],pc[:,2],'.',markersize=1)
    ax.set_xlim([-3.2,3.2])
    ax.set_ylim([-3.2,3.2])
    ax.set_zlim([-3.2,3.2])
    plt.title('plane with line')
    plt.show()
                
    # 平面检测，<p,n>=D
    n,D=plane_det_pc_ransac(pc,r=0.2,k=0.3,th=0.1)
    print('nx,ny,nz:',n)
    print('D',D)
    
    # 计算点到平面的距离，用于选出平面上的点
    dist=pc_to_plane_dist(n,D,pc)
    pc_sub=pc[dist<0.1]
    # 显示原始点云，及平面对应的点云
    ax=plt.figure().gca(projection='3d')
    ax.plot(pc[:,0],pc[:,1],pc[:,2],'.',markersize=1)
    ax.plot(pc_sub[:,0],pc_sub[:,1],pc_sub[:,2],'r.',markersize=1)
    ax.set_xlim([-3.2,3.2])
    ax.set_ylim([-3.2,3.2])
    ax.set_zlim([-3.2,3.2])
    plt.title('detected plane')
    plt.show()
    
    # 显示扣除了平面上的点
    pc_sel=pc[dist>=0.1]
    ax=plt.figure().gca(projection='3d')
    ax.plot(pc_sel[:,0],pc_sel[:,1],pc_sel[:,2],'g.',markersize=1)
    ax.set_xlim([-3.2,3.2])
    ax.set_ylim([-3.2,3.2])
    ax.set_zlim([-3.2,3.2])
    plt.title('line without plane')
    plt.show()
    
    # 直线检测
    d,m=line_det_pc_ransac(pc_sel,r=0.2,k=0.2,th=0.15)
    print('dx.dy.dz:',d)
    print('mx,my,mz:',m)
    
    # 计算点到直线距离（用于分理处支线上的点云）
    dist=pc_to_line_dist(d,m,pc)
    # 显示原始点云以及检测的直线点云
    pc_sub=pc[dist<0.1]
    ax=plt.figure().gca(projection='3d')
    ax.plot(pc[:,0],pc[:,1],pc[:,2],'.',markersize=1)
    ax.plot(pc_sub[:,0],pc_sub[:,1],pc_sub[:,2],'r.',markersize=1)
    ax.set_xlim([-3.2,3.2])
    ax.set_ylim([-3.2,3.2])
    ax.set_zlim([-3.2,3.2])
    plt.title('detected line')
    plt.show()
    

