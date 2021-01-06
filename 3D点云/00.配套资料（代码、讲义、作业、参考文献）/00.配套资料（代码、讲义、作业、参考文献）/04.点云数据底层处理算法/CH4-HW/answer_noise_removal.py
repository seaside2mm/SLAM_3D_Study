import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt5Agg')
# 逐点根据K近邻最大距离过滤

# 加载原始点云数据
pc=np.genfromtxt('/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/04.点云数据底层处理算法/CH4-HW/pc.csv', delimiter=',').astype(np.float32)


def test_knn_cv(pc):
    import cv2
    # algorithm=1: KD-tree, trees=5:构建5棵树，加快搜索(1-16), checks=50: 查询递归次数
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=16), dict(checks=50))

    # KNN搜索
    print('KNN (openCV)')
    matches = flann.knnMatch(pc,pc,k=10)                        
    
    print('filtering...')
    pc_new=np.array([pc[i] for i,m in enumerate(matches) if max([n.distance for n in m])<0.005])
    return pc_new


def test_knn_pcl(pc):
    import pcl
    cloud = pcl.PointCloud(pc.astype(np.float32))
    
    # 构建kdtree用于快速最近邻搜索
    kdtree = cloud.make_kdtree_flann()
    
    # K近邻搜索
    print('KNN (PCL)')
    [idx, dist] = kdtree.nearest_k_search_for_cloud(cloud, 10)

    print('filtering...')
    pc_new=pc[np.max(dist,axis=1)<0.005**2]
    return pc_new

def plot(pc, pc_new, choice=1):
    
    if choice == 1:
        # 装载私有库
        import sys
        sys.path.append('../../code')
        from depth_img_view_cv import view_pc_3d

        CAM_WID,CAM_HGT = 640,480
        CAM_FX,CAM_FY   = 795.209,793.957
        CAM_CX,CAM_CY   = 332.031,231.308

        IMG_WID,IMG_HGT = CAM_WID,CAM_HGT

        view_pc_3d(pc,cam_fx=CAM_FX,cam_fy=CAM_FY,\
                    cam_cx=CAM_CX,cam_cy=CAM_CY,\
                    cz=1.1, \
                    img_wid=IMG_WID,img_hgt=IMG_HGT,\
                    dmin=0.5,dmax=1.5)

        view_pc_3d(pc_new,cam_fx=CAM_FX,cam_fy=CAM_FY,\
                    cam_cx=CAM_CX,cam_cy=CAM_CY,\
                    cz=1.1, \
                    img_wid=IMG_WID,img_hgt=IMG_HGT,\
                    dmin=0.5,dmax=1.5)
    else:

        print('ploting...')
        ax = plt.figure().gca(projection='3d')
        ax.plot(pc[:,0],pc[:,1],pc[:,2],'.',markersize=1)
        plt.title('pc')
        plt.show()

        ax = plt.figure().gca(projection='3d')
        ax.plot(pc_new[:,0],pc_new[:,1],pc_new[:,2],'.',markersize=1)
        plt.title('pc_new')
        plt.show()
if __name__ == "__main__":
    pc_new = test_knn_cv(pc)
    plot(pc,pc_new,choice=2)