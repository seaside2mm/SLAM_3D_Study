import  numpy as np


# Laplacian平滑

# 移动最小二乘法
# pc_nn:pc的最近邻点集
def mls(p, pc_nn):
    num = pc_nn.shape[0]

    # 计算近邻的加权平面拟合
    pc_w  = pc_nn #*dist.reshape(num,1)
    pc_w = pc_w- np.mean(pc_w, axis = 0)
    
    # 用特征值分解计算平面方向
    M = np.dot(pc_w.T, pc_w)
    E, F = np.linalg.eig(M)
    idx  = np.argsort(E)
    uz = F[:, idx[0]].ravel()  # 法向量方向，对应最小特征值
    ux = F[:, idx[1]].ravel()
    uy = F[:, idx[2]].ravel()
    
def mls():
    # 计算目标点到平面的投影
    px, py, pz = np.sum(p*ux), np.sum(p*uy), np.sum(p*uz)
    
    #计算所有邻近点到平面的投影，以px, py为中心平移
    pc_nnx = np.dot(pc_nn, ux) - px   #用內积计算领域点在平面上投影
    pc_nny = np.dot(pc_nn, uy) - py   #投影点以待滤波器的投影位置为原点
    pc_nnz = np.dot(pc_nn, uz) - pz
    
    #计算两次拟合函数，系数w
    pc_nnx.shape = pc_nny.shape=pc_nnz.shape=num
    V = np.array([np.ones(num), pc_nnx, pc_nny, pc_nnx * pc_nny, pc_nnx ** 2, pc_nny ** 2]).T
    w = np.dot(np_lin.inv(np.dot(V.T,V)),np.dot(V.T,pc_nnz))

    #计算拟合函数在原点数值，沿着法向量方向调整作为目标点平滑结果
    p_new = p+(w[0]-pz)*uz #沿着法向量方向移动p点
    return p_new

