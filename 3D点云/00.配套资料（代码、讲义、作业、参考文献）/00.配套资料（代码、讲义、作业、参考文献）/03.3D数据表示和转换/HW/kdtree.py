import numpy as np
import heapq

# 先序遍历递归构建kd树（kd_node : kd_node1, kd_node2, 分割点坐标）
# points需要分割的点集，dim 总的维度，i  对应当前切分的维度
def make_kd_tree(points, dim, i=0):
    if len(points) > 1:
        # 按照维度进行排序
        points = points[np.argsort(points[:,i]), :]
        i = (i+1) % dim    
        half = (len(points)-1) // 2  #切分点选择那些尽量能够将点平分在两个子空间的点
        return (make_kd_tree(points[: half, :], dim, i),     # left 
                make_kd_tree(points[half + 1, :], dim, i),   # right
                points[half, :])
        
    elif len(points) == 1:
        return (None, None, points[0, :])  # 叶子节点，只有一个点

# 最近邻搜索
# 返回最优点距离与坐标
def get_nearest(kd_node, point, dim, dist_func, return_distances=False, i=0, best=None):
    if kd_node:
        dist = dist_func(point, kd_node[2])  # 到分割点距离
        dx = kd_node[2][i] - point[i]        # 到分割平面距离
        # 保存最优结果
        if not best:   # best存放找到的最近邻
            best = [dist, kd_node[2]]  # 搜索刚开始
        elif dist < best[0]:
            best[0], best[1] = dist, kd_node[2]
        i = (i + 1) % dim
        
        #根据相对分割平面的位置，选择搜索一侧子树
        get_nearest(kd_node[dx<0], point, dim, dist_func, return_distances, i, best)

        # 如果到分割平面距离平方（dx**2)  小于当前最小距离平方(best[0]), 则需要搜索另一侧树
        if dx**2 < best[0]:
            get_nearest(kd_node[dx >=0], point, dim, dist_func, return_distances, i, best)
    return best if return_distances else best[1]
        
# knn搜索， 使用了优先队列
# 注意，k大于节点数一半时，可能露点
def get_knn(kd_node, point, k, dim, dist_func, return_distances=False, i=0, heap=None):
    is_root = not heap
    if is_root: heap = []
    
    if kd_node:
        dist = dist_func(point, np.array(kd_node[2]))  #节点距离（进入队列时将array变成了list，需要恢复）到分割点距离
        dx =  kd_node[2][i] - point[i]  #相对分割平面的位置
        if len(heap) < k:   
            heapq.heappush(heap, (-dist, kd_node[2].tolist())) #直接入队（heapq不能直接处理array)
        elif dist < -heap[0][0]: #替换队列中最远点
            heapq.heappushpop(heap, (-dist, kd_node[2].tolist()))

        i = (i + 1) % dim
        
        #根据相对分割平面的位置，选择搜索一侧子树
        get_knn(kd_node[dx < 0], point, k, dim, dist_func, return_distances, i, heap)
        
        # 如果到分割平面的距离平方小于当前近邻集合中最大距离平方(-heap[0][0]),则需要搜索另一边
        if dx**2 < -heap[0][0] or len(heap) < k:  
            get_knn(kd_node[dx >= 0], point, k, dim, dist_func, return_distances, i, heap)
            
    if is_root: #搜索完成
        idx = np.argsort([-h[0] for h in heap])
        neighbors = [(-heap[n][0], np.array(heap[n][0])) for n in idx]

        return neighbors if return_distances else [n[1]] for n in neighbors


def dist_func(pointA, pointB):
    return np.sum(np.square(pointA - pointB))

if __name__ == "__main__":
    points = np.array([[1,5],[2,5],[3,5],[4,5]])
    kd_node = make_kd_tree(points, 2)
    
    # print(dist_func(np.array([0,0,0]), p.array([1,1,1])))
    
    # search_point = np.array([0,0,0.3])
    # get_nearest(kd_node, search_point, 3, dist_func)
                      
            