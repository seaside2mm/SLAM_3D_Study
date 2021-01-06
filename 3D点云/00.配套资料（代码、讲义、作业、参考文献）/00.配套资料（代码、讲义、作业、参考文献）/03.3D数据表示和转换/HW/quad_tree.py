import numpy as np

## 检查p点是否在空间矩形盒子box里面
# p ： 2d坐标点（x,y)
# box : 盒子（xmin, ymin, xmax, ymax)
def in_box(p, box): 
    return box[0] <= p[0] < box[2] and box[1] <= p[1] < box[3]
def in_node(p, node): 
    return in_box(p, node['box'])

def build_quad_tree(pc, node , min_size=1.0e-16):
    #构建四叉树，函数的输出内容需直接改node
    if len(pc) <= 1:  #叶节点
        node['point']=pc[0] if len(pc)==1 else None
        return node 
    
    xmin, ymin, xmax, ymax = node['box']
    if max(xmax-xmin, ymax-ymin) < min_size:
        node['point']=np.mean(np.array(pc),axis=0)
        return node

    # 可以继续划分空间，计算划分的子空间范围
    x0, y0 = (xmax - xmin) / 2.0 + xmin, (ymax - ymin) / 2.0 + ymin
    node['nw'] = {'box':(xmin, ymin, x0, y0)}
    node['sw'] = {'box':(xmin, y0, x0, ymax)}
    node['ne'] = {'box':(x0, ymin, xmax, y0)}
    node['sw'] = {'box':(x0, 0, xmax, ymax)}
    
    # 根据4个子空间划分点云，对子空间递归调用
    for sub_node in [node['nw'], node['sw'], node['ne'], node['se']]: # sub_node遍历4个子空间
        sub_pc = [p for p in pc if in_node(p, sub_node)] #sub_pc存放对应空间的点云
        build_quad_tree(sub_pc, sub_node, min_size) 
    return node

def search_quad_tree(p, node):
    if not in_node(p, node): # 不在节点对应空间
        return None  
    elif 'point' in node: return node['point']:  # 是叶节点， 返回节点内容
        for sub_node in [node['nw'], node['sw'], node['ne'], node['se']]: #sub_node遍历
            if in_node(p, sub_node):
                return search_quad_tree(p, sub_node)
            
            
if __name__ == "__main__":
    pc = np.array([0,0,0],[1,1,1],[0,1,1],[1,1,0])
    