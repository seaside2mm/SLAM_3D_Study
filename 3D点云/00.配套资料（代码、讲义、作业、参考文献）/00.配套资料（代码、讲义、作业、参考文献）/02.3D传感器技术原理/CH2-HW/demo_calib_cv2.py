import numpy as np
import cv2
import matplotlib.pylab as plt

####################
# 相机参数标定
# 参考： https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
####################

import numpy as np
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 棋盘格的每个格点的列号和行号
# (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images =['%d.jpg'%d for d in range(1,9)]

for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 角点粗检测
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    if not ret: 
        continue
    else:
        objpoints.append(objp)    
    
    # 角点精检测
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)

    # 绘图
    if True:
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        plt.imshow(img)
        plt.title(fname)
        plt.show()

print('objpoints:',objpoints)
print('imgpoints:',imgpoints)
print('corners:',corners)
print('corners2:',corners2)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print('mtx:',mtx)       # 内参矩阵
print('dist:',dist)     # 失真参数
print('rvecs:',rvecs)
print('tvecs:',tvecs)

## 测试失真矫正
fname=images[3]
img = cv2.imread(fname)
h,  w = img.shape[:2]
print('img.shape:',img.shape)

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
print('newcameramtx:',newcameramtx)
print('roi:',roi)

# undistort method 1
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
plt.subplot(2,1,1)
plt.imshow(img)
plt.title('original')
plt.subplot(2,1,2)
plt.imshow(dst)
plt.title('corrected 1')
plt.show()
        
# undistort method 2
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
print('mapx:',mapx)
print('mapy:',mapy)

x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
plt.subplot(2,1,1)
plt.imshow(img)
plt.title('original')
plt.subplot(2,1,2)
plt.imshow(dst)
plt.title('corrected 2')
plt.show()

# undistort method 3
dst = cv2.undistort(img,mtx,dist)
plt.subplot(2,1,1)
plt.imshow(img)
plt.title('original')
plt.subplot(2,1,2)
plt.imshow(dst)
plt.title('corrected 3')
plt.show()


