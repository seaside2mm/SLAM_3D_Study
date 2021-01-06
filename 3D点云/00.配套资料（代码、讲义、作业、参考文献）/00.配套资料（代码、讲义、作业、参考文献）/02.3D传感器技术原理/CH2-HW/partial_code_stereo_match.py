####################
# SAD双目匹配算法
# 
# 左右图像中同一物体对应的纵坐标相同，仅仅横坐标不同，
# 并且右图像中物体“偏左”移动。
# 假设左右图像对应像素的水平偏移不超过30像素
# 
# matchLR用SAD算法计算视差图: Δ𝑑 = 𝑎𝑟𝑔𝑚𝑖𝑛𝑓(𝑥, 𝑦, 𝑑)
# 视差计算距离Z的公式: Z = (B*f)/Δ𝑑 

# FINISHED IN 2020.10.11 BY SEASIDE
# TAKES ABOUT One HOURS
####################

import numpy as np
import cv2
import matplotlib.pyplot as plt
def read_file(show=True):
    # 从CSV读取灰度图imgL和imgR
    print('loading image from CSV file')
    imgL = np.genfromtxt('aL_gray.csv',delimiter=',').astype(np.float32)
    imgR = np.genfromtxt('aR_gray.csv',delimiter=',').astype(np.float32) 
    
    if(show):
        plt.clf()
        plt.subplot(1,2,1);plt.imshow(imgL,cmap='gray')
        plt.subplot(1,2,2);plt.imshow(imgR,cmap='gray')
        plt.show()
    return [imgL, imgR]

def matchLR(imgL, imgR, show=True):
    ## 在下面补充你的代码，对imgL中的每个像素，找到imgR中匹配的像素，
    ## 并将匹配像素的水平偏移量（取绝对值）保存在文件math.csv中
    D=40    # 像素匹配搜索是，最大偏移量
    WIN_SIZE=7     # 像素匹配搜索时，窗口大小
    H, W = imgL.shape
    #构建一系列平移后的图img_shift
    img_shift = np.zeros((D, H, W))
    for d in range(D):
        img_shift[d,:,:] = np.roll(imgR, d, axis=1)
    
    # 计算左图和一系列平移后的右图的差，取绝对值
    img_diff = np.abs(img_shift - imgL)
    
    #对图像差计算窗口平滑
    for n in range(img_diff.shape[0]):
        img_diff[n,:,:] = cv2.boxFilter(img_diff[n,:,:], -1, (WIN_SIZE,WIN_SIZE))
    
    # 逐个像素求最匹配的平移量
    imgD = np.zeros((H, W))
    imgD = np.argmin(img_diff, axis=0)
    ## 下面是保存CSV代码的例子
    # data=np.random.randint(0,10,(427,370))  # 生成尺寸为427x370的随机整数矩阵
    np.savetxt('match.csv', imgD, fmt='%d', delimiter=',', newline='\n') # 保存为csv文件
    if(show):
        plt.imshow(imgD)
        plt.show()
    return imgD


if __name__ == '__main__':
    
    imgL, imgR = read_file(False)
    imgD = matchLR(imgL, imgR, True)

 

