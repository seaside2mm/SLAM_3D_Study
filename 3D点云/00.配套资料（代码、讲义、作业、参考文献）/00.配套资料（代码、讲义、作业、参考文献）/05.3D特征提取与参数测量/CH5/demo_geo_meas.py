import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D

####################
# 刚体变换， 选择，平移
####################

# 依次绕三个轴旋转对应的旋转矩阵可以通过将三个旋转矩阵相乘的到
def get_rot_mat(ax=0,ay=0,az=0):
    Rx = np.array([[1,                0,                    0],
                   [0,       np.cos(ax),           np.sin(ax)],
                   [0,      -np.sin(ax),           np.cos(ax)]])
                  
    Ry = np.array([[np.cos(ay),       0,          -np.sin(ay)],
                   [0,                1,                    0],
                   [np.sin(ay),       0,           np.cos(ay)]])
                  
    Rz = np.array([[np.cos(az),     np.sin(az),         0],
                   [-np.sin(az),    np.cos(az),         0],
                   [0,                       0,         1]])
                  
    return np.dot(np.dot(Rx,Ry),Rz)

####################
# 几何参数提取， 法向量
# 形状拟合，直线，平面，圆， 球
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
#   PCA计算平面模型参数：<n,p>=D
#   p对于原点的向量在法向量上的投影D
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
    return n,D,m

## 功能描述：
#   平面拟合，最小二乘法
#   (V^TV)^{-1}V^T
# 返回
#   n：  平面法向量方向
#   D：  原点到平面的距离（沿着法向量方向，可能是负值）
def plane_det_pc_2(pc):
    return n,D,m

## 功能描述：
#   平面拟合，矢量叉乘，适合无噪声条件
#   v = （a-c)x(a-c)   =》   n = v/||v||
# 返回
#   n：  平面法向量方向
#   D：  原点到平面的距离（沿着法向量方向，可能是负值）
def plane_det_pc_3(pc):
    m = np.mean(pc,axis=0)    # 平面上一点（点云中心）
    n = np.linalg.norm(np.cross(pc[0]-m,pc[100]-m))
    D=np.sum(n*m)           # 平面上的点m在法向量上投影长度
    return n,D,m


# 计算法向量，数据是以传感器图像顺序排列的点云
def calc_img_pc_norm(img_pc):
    img_pc1 = img_pc - np.roll(img_pc, 1, axis=0)  #最外界数据去除不用  
    img_pc2 = img_pc - np.roll(img_pc, 1, axis=1)
    
    img_norm = np.cross(img_pc1, img_pc2)  
    return img_norm/np.linalg.norm(img_norm,axis=2)[:,:,np.newaxis] #法向量归一化


# 平面参数计算——噪声条件下，用KNN加PCA检测平面法向量
def calc_pc_norm_knn(img_pc, k=64):
    pc = img_pc.reshape(-1,3).astype(np.float)
    
    #构建kdtree用于最近邻
    import pcl 
    cloud = pcl.PointCloud(pc)
    kdtree = cloud.make_kdtree_flann()
    [idx, dist] = kdtree.nearest_k_search_for_cloud(cloud, k)
    
    def calc_pc_norm_eig(pc):
        m=np.mean(pc,axis=0)    # 平面上一点（点云中心）
        pc0=pc-m                # 去均值
        C=np.dot(pc0.T,pc0)     # 协方差阵
        E,V=np.linalg.eig(C)    # E: 特征值, V: 特征向量
        return V[:,np.argmin(E)].ravel()        # 最小特征值对应直线方向
        
    #对每个点根据近邻计算法向量方向
    img_norm = np.array([calc_pc_norm_eig(pc[i,:]) for i in idx])
    
    # 法向量统一方向
    img_norm[img_norm[:,:,2]>0, :] *= -1
    
    return img_norm


# 多项式曲面参数拟合（LS）
# SVD求解
def todo():
    pass


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
    b=np.sum(pc**2,axis=1).reshape(-1,1)  # xi^2+yi^2+zi^2
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


####################
# 几何计算
####################

# 点在向量上投影  v' = <v,d> dot d/||d||^2 
def pnt_to_vec_proj(v,d):
    return np.dot(np.dot(v,d),d/(np.linalg.norm(d)**2))
    

## 功能描述：
#   计算点v在直线上的投影坐标  v' = m + <v-m,d> dot d/||d||^2 
# 直线参数方程：p=dt+m
def pnt_to_line_proj(v,d,m=0):
    return m+np.sum((v-m)*d)*d/np.sum(d**2)


## 功能描述：
#   计算点v在平面上的投影坐标
#   1. 先计算v在向量n上的投影到平面的距离 T = <v,n> -D
#   2. 计算v沿着n相反方向移动距离t得到的坐标 v' = v- nt
# 平面方程是<p,n>=D
def pnt_to_plane_proj(v,n,D):
    n0=n/np.linalg.norm(n)  # 长度归一化
    return v-n0*(np.sum(n*v)-D)


## 功能描述：
#   计算3D空间点到直线的距离 d = |n x m| / |n|
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


####################
# 存在离群点条件下的点云几何模型参数拟合
####################


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
    
"""  
点云形状拟合——RANSAC对多个集合形状的拟合
1. 使用RANSAC抽取第一个几何体的参数
2. 删除第一个几何体对应的点云
3. 使用RANSAC抽取第二个几何体参数
4. 注意：这一算法依靠点云随机采样，有可能失败
"""
def plane_det_pc_ransac(pc,r=0.2,k=0.4,th=0.06,it=20, it1=3, verbose=False):
    N=pc.shape[0]
    M,K=int(N*r),int(N*k)
    
    idx=np.arange(N)
    while it>0:
        if verbose: print('iteration',it)
        it-=1
        np.random.shuffle(idx)
        pc_sub=pc[idx[:M]]
        n,D,_=plane_det_pc(pc_sub)
        dist=pc_to_plane_dist(n,D,pc)
        if np.sum(dist<th)>K: break

    if it<=0: 
        print('RANSAC fail')
        return (None,None)
    
    if verbose: print('matched')
    while it1>0: #微调
        it1-=1 
        pc_sub=pc[dist<th]
        n,D,_=plane_det_pc(pc_sub)
        dist=pc_to_plane_dist(n,D,pc)
        
    return n,D  

####################
# 3D点云配准
####################

# min||(pc1-m1)*R0-(pc0-m0)||
def icp(pc0, pc1,it=6,verbose=True):
    import cv2
    #使用opencv的KNN查询，也可以换成使用PCL实现)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=50))
    #去均值
    m0 = np.mean(pc0, axis=0)
    m1 = np.mean(pc1, axis=0)
    p = (pc0-m0).astype(np.float32)
    q = (pc1-m1).astype(np.float32)

    R0 = np.eye(3)  #存放总的旋转矩阵
    for n in range(it):
        if True:
            #对p的每一点，找到在q的对应点，最近的点
            q_match = np.array([q[m[0].trainIdx] for m in flann.knnMatch(p,q,k=1)])
            q_match -= np.mean(q_match, axis=0)

            #计算旋转矩阵
            H = np.dot(q_match.T, p)
            u, _, vh = np.linalg.svd(H)
            R = np.dot(u,vh)

            #调整点云位置
            q = np.dot(q,R)
            R0 = np.dot(R0, R)
        if False:
            #对q的每一点，找到在p的对应点，最近的点
            p_match = np.array([p[m[0].trainIdx] for m in flann.knnMatch(q,p,k=1)])
            p_match -= np.mean(p_match, axis=0)

            #计算旋转矩阵
            H = np.dot(p_match.T, q)
            u, _, vh = np.linalg.svd(H)
            R = np.linalg.inv(np.dot(u,vh))  #取逆，需要作用在q点云上，而不是p

            #调整点云位置
            q = np.dot(q,R)
            R0 = np.dot(R0, R)
        if verbose: print('[%d] err'%n,np.mean(np.abs(p-q)))
    return R0, m0, m1

####################
# 从点云重建3D物体表面
####################

def pc_ext(pc, p0, d, k):
    pcn = find_surf_dir_with_knn(pc, k, func_phi)
    
    # 法向量方向修正，这里简化，仅要求法向量背离给定的中心点p0
    pcn = np.array([n*np.sign(np.dot(n,p-p0)) for p,n in zip(pc,pcn)])
    pc_out = pc+pcn*d
    pc_in = pc-pcn*d

    return pc_in, pc_out

## 构建SDF RBF
# p0 内点中心，用于确定法向量方向
# d 点云沿着法向量正负方向拓展距离
# k 计算法向量knn数量
def contruct_SDF(pc, p0, d, k):
    # 点云安法向量方向拓展
    pc_in, pc_out = pc_ext(pc, p0, d, k)
    pc_all = np.vstack((pc, pc_out, pc_in))
    #解线性方程计算
    M = np.array([func_phi(np.linalg.norm(pc_all[r]-pc_all,axis=1)) for r in range(3*N)])
    v = np.array([np.zeros(N), D*np.ones(N), -D*np.ones(N)]).ravel()
    w = np.dot(np.linalg.pinv(M), v).ravel()  
    return pc_all, w




def test_line():

    ###############################
    # 加载直线点云
    pc=np.genfromtxt('/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/05.3D特征提取与参数测量/CH5/pc_line.csv', delimiter=',').astype(np.float32)
    
    # 直线点云参数提取：p=d*k+m
    d,m=line_det_pc(pc)
    print('dx.dy.dz:',d)
    print('mx,my,mz:',m)
    
    # 显示直线检测结果
    pc_line=np.array([d*r+m for r in np.linspace(-2,2,100)])
    ax=plt.figure().gca(projection='3d')
    ax.plot(pc[:,0],pc[:,1],pc[:,2],'.',markersize=1)
    ax.plot(pc_line[:,0],pc_line[:,1],pc_line[:,2],'r.',markersize=1)
    ax.set_xlim([-2.2,2.2])
    ax.set_ylim([-2.2,2.2])
    ax.set_zlim([-2.2,2.2])
    plt.title('line detection')
    plt.show()
    
def test_plane():
    pc=np.genfromtxt('/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/05.3D特征提取与参数测量/CH5/pc_plane.csv', delimiter=',').astype(np.float32)

    # 平面的点云参数提取，<p,n>=D
    n1,D1,m1 =plane_det_pc(pc)
    n3,D3,m3 =plane_det_pc_3(pc)
    
    pc_line1=np.array([n1*r+m1 for r in np.linspace(-2,2,100)])
    pc_line3=np.array([n3*r+m3 for r in np.linspace(-2,2,100)])
    
    # 显示加载的平面点云
    ax=plt.figure().gca(projection='3d')
    ax.plot(pc[:,0],pc[:,1],pc[:,2],'.',markersize=1)
    ax.plot(pc_line1[:,0],pc_line1[:,1],pc_line1[:,2],'r.',markersize=1)
    ax.plot(pc_line3[:,0],pc_line3[:,1],pc_line3[:,2],'b.',markersize=1)
    ax.set_xlim([-3.2,3.2])
    ax.set_ylim([-3.2,3.2])
    ax.set_zlim([-3.2,3.2])
    plt.title('plane')
    plt.show()
    
    # 计算点到平面的距离，用于选出平面上的点
    dist=pc_to_plane_dist(n,D,pc)
    print('mean distance:',np.mean(np.abs(dist)))

def test_obj():
    pc=np.genfromtxt('/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/05.3D特征提取与参数测量/CH5/pc_obj.csv', delimiter=',').astype(np.float32)
    
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
    # ax=plt.figure().gca(projection='3d')
    # ax.plot(pc[:,0],pc[:,1],pc[:,2],'.',markersize=1)
    # ax.plot(pc_sub[:,0],pc_sub[:,1],pc_sub[:,2],'r.',markersize=1)
    # ax.set_xlim([-3.2,3.2])
    # ax.set_ylim([-3.2,3.2])
    # ax.set_zlim([-3.2,3.2])
    # plt.title('detected plane')
    # plt.show()
    
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
    
def test_icp():
    np.random.seed(seed=1)
    # 测试数据
    x = np.arange(0, 10)
    y = 2 * x + 3
    z = 3 * x 
    pc0 = np.column_stack([x, y,z]) 
    R = get_rot_mat(ax=30,ay=30)
    pc1 = np.transpose(np.dot(R,pc0.T))
    #icp匹配后
    R0 = icp(pc0, pc1,verbose=False)
    R0 = R0[0]
    pc2 = np.transpose(np.dot(R0,pc0.T))
    
    ax=plt.figure().gca(projection='3d')
    ax.plot(pc0[:,0],pc0[:,1],pc0[:,2],'b.',markersize=1)
    ax.plot(pc1[:,0],pc1[:,1],pc1[:,2],'r.',markersize=1)
    ax.plot(pc2[:,0],pc2[:,1],pc2[:,2],'g.',markersize=1)
    plt.show()
    
def test_sdf():
    np.random.seed(seed=1)
    # 测试数据
    x = np.arange(0, 10)
    y = 2 * x + 3
    z = 3 * x 
    pc0 = np.column_stack([x, y,z]) 
    
    #sdf
    pc_all = contruct_SDF(pc0,[0,0,0],1,3)
    
    ax=plt.figure().gca(projection='3d')
    ax.plot(pc_all[:,0],pc_all[:,1],pc_all[:,2],'r.',markersize=1)
    plt.show()
    
if __name__ == '__main__':
    test_line()
    
