import pcl
import numpy as np

np.random.seed(1234)
cloud = pcl.PointCloud()
cloud.from_array(np.random.rand(100,3).astype(np.float32))

octree = cloud.make_octreeSearch(0.2)
octree.add_point_from_input_cloud()

searchPoint = pcl.PointCloud()
searchPoint.from_array(np.random.rand(5,3).astype(np.float32))

K = 10
print("\n#################### KNN with K: %d ################3"%K)
[idx, dist] = octree.nearest_K_search_from_cloud(searchPoint, K)
for idx0, dist0, pnt in zip(idx, dist, searchPoint):
    print("search center:",pnt)
    for i,d in zip(idx0, dist0):
        x,y,z = cloud[i]
        print("  id:%d, dist:%.4f,  (%.4f,%.4f,%.4f)"%(i,d**0.5,x,y,z))