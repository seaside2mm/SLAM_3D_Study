import numpy as np

# 滑动窗口形式计算滤波
kernal = np.array([[1.0, -0.5, 1.0],
                  [-0.5, 2.0, -0.5],
                  [1.0, -0.5, 1.0]]).astype(np.float32)

# imgf = cv2.filter2D(img_dep, cv2.CV_32F, ker) #opencv实现
kernal /= np.sum(kernal)   #卷核归一化
imgf = img_dep.copy()

for y in range(1, IMG_HGT-1):
    for x in range(1, IMG_WID-1):
        win_dep = img_dep[y-1:y+2, x-1:x+2]
        imgf[y,x] = np.sum(kernal*win_dep)
        
        
# 双边滤波,距离加权项,差异加权项
# 𝐼_{out}(𝑝) = 1/𝐾𝑝*sum(ℎ(||𝑝 − 𝑞||)*𝑤(𝑝, 𝑞)*𝐼_{in}(𝑞))

# 对深度图
# 滤波半径（滑动窗口尺寸, 深度差异对应的权重衰减程度, 像素距离的权重衰减程度
# imgf = cv2.bilateralFilater(img_dep, 5,1,1,1)
ker1 = np.ones((2*W, 2*W))
for y in range(W, IMG_HGT-W):
    for x in range(W, IMG_WID-W):
        win_dep = img_dep[y-W:y+W, x-W:x+W]
        ker2 = np.exp(-(win_dep-img_dep[y,x])**2/0.02)
        
        ker = ker1 * ker2
        ker /= np.sum()
        imgf[y,x] = np.sum(ker * win_dep)
    
# 对激光强度图    
ker1 = np.ones((2*W, 2*W))
for y in range(W, IMG_HGT-W):
    for x in range(W, IMG_WID-W):
        win_dep = img_dep[y-W:y+W, x-W:x+W]
        win_amp = img_amp[y-W:y+W, x-W:x+W]
        ker2 = np.exp(-(win_amp-img_amp[y,x])**2/10.0)
        
        ker = ker1 * ker2
        ker /= np.sum()
        imgf[y,x] = np.sum(ker * win_dep)
        
# 时域IIR滤波

