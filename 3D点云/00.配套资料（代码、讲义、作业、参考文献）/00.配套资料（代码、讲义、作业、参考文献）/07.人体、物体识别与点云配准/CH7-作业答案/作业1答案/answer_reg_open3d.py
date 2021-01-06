
import numpy as np
import open3d as o3d
import copy

# numpy转点云格式
def np_to_pcd(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    return pcd

# 点云格式转numpy
def pcd_to_np(pcd): return np.asarray(pcd.points)

# 预处理
def preprocess_point_cloud(pcd, voxel_size):
    # 降采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # 估计法向量方向
    radius_normal = voxel_size*2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # 计算FPFH特征
    radius_feature = voxel_size*5
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(10000000, 500))
    return result


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def show_result():
    # 显示结果
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    pc_src=pcd_to_np(source)
    pc_tgt=pcd_to_np(target)
    R=result_ransac.transformation
    pc_src_trans=np.dot(pc_src,R[:3,:3].T)+R[:3,3].T
    
    np.savetxt('/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/07.人体、物体识别与点云配准/CH7-作业答案/作业1答案/out.csv', pc_src_trans, fmt='%.8f', delimiter=',', newline='\n')
    
    pc_out  =np.genfromtxt('/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/07.人体、物体识别与点云配准/CH7-作业答案/作业1答案/out.csv',delimiter=',')
    pc_scene=np.genfromtxt('/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/07.人体、物体识别与点云配准/CH7-作业答案/作业1答案/scene.csv',delimiter=',')
    
    ax=plt.figure().gca(projection='3d')
    ax.plot(pc_scene[:,0],pc_scene[:,1],pc_scene[:,2],'.b',markersize=0.5)
    ax.plot(pc_out  [:,0],pc_out  [:,1],pc_out  [:,2],'.r',markersize=0.5)
    plt.show()
    
def main():
    voxel_size = 5
    
    # 加载数据并作格式转化
    source=np_to_pcd(np.genfromtxt('/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/07.人体、物体识别与点云配准/CH7-作业答案/作业1答案/ds.csv',delimiter=','))
    target=np_to_pcd(np.genfromtxt('/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/07.人体、物体识别与点云配准/CH7-作业答案/作业1答案/scene.csv',delimiter=','))
    
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)


    # 配准        
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    draw_registration_result(source_down, target_down, result_ransac.transformation)    
    
    #show_result()
    

if __name__ == "__main__":
    main()
    