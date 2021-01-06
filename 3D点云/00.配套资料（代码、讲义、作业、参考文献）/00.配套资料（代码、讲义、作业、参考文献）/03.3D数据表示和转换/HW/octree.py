


## 检查p点是否在空间矩形盒子box里面
# p ： 3d坐标点（x,y,z)
# box : 盒子（xmin, ymin, zmin, xmax, ymax,zmax)
def in_box(p, box): 
    return box[0] <= p[0] < box[3] and \
           box[1] <= p[1] < box[4] and \
           box[2] <= p[1] < box[5]
def in_node(p, node): 
    return in_box(p, node['box'])

#构建四叉树，函数的输出内容需直接改node
def build_octree(pc, node , min_size=1.0e-16):
    if len(pc) <= 1:  #叶节点
        node['point']=pc[0] if len(pc)==1 else None
        return node 
    
    xmin, ymin, zmin, xmax, ymax, zmax = node['box']
    if max(xmax-xmin, ymax-ymin, zmin-zmax) < min_size:
        node['point']=np.mean(np.array(pc),axis=0)
        return node
    
    if len(pc) > 1:
    # 可以继续划分空间，计算划分的子空间范围
    x0 = (xmax - xmin) / 2.0 + xmin  #空间切分点
    y0 = (ymax - ymin) / 2.0 + ymin
    z0 = (zmax - zmin) / 2.0 + zmin
    
    node['n0'] = {'box': (xmin, ymin, zmin ,xmax, ymax, zmax)}
    node['n1'] = {'box': (x0, ymin, zmin ,xmax, y0, z0)}
    node['n2'] = {'box': (xmin, ymin, z0 ,x0, y0, zmax)}
    node['n3'] = {'box': (x0, ymin, z0 ,xmax, y0, zmax)}
    node['n4'] = {'box': (xmin, y0, zmin ,x0, ymax, z0)}
    node['n5'] = {'box': (x0, y0, zmin ,xmax, ymax, z0)}
    node['n6'] = {'box': (xmin, y0, z0 ,x0, ymax, zmax)}
    node['n7'] = {'box': (x0, y0, z0 ,xmax, ymax, zmax)}
    
    # 根据8个子空间划分点云，对子空间递归调用
    for sub_node in [node['n%d'%d] for d in range(8)]: # sub_node遍历8个子空间
        sub_pc = [p for p in pc if in_node(p, sub_node)] #sub_pc存放对应空间的点云
        build_quad_tree(sub_pc, sub_node, min_size) 
    return node

def search_octree(p, node):
    if not in_node(p, node): # 不在节点对应空间
        return None  
    elif 'point' in node:  # 是叶节点， 返回节点内容
        return node['point']  
    else:   # 非叶节点，检查是否在8个子空间任何一个
        for sub_node in [node[n%d]%d for d in range(8)]: #sub_node遍历
                if in_node(p, sub_node):
                    return search_octree(p, sub_node)
    