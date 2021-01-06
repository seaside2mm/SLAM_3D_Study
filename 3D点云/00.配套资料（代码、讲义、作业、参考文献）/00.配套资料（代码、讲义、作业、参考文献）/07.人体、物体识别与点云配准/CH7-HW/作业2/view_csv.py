
import numpy as np
from pc_view import pc_view

CAM_HGT,CAM_WID=424,512
CAM_CX=2.543334e+02
CAM_CY=1.967047e+02
CAM_FX=3.664861e+02
CAM_FY=3.664861e+02
CAM_F=(CAM_FX+CAM_FY)*0.5

IMG_HGT,IMG_WID=CAM_HGT,CAM_WID
IMG_SZ =IMG_WID*IMG_HGT
IMG_SHAPE=(IMG_HGT,IMG_WID)

for fname_csv in ['/Users/seaside/Desktop/【第02部分 三维点云专题】三位建模感知与点云视频教程 5套/03.2020年3D感知技术培训视频教程（附代码、讲义、参考资料）49课/00.配套资料（代码、讲义、作业、参考文献）/00.配套资料（代码、讲义、作业、参考文献）/07.人体、物体识别与点云配准/CH7-HW/作业2/data_csv/pc%d.csv'%n for n in range(0,500,10)]:    
    print(fname_csv)
    pc=np.genfromtxt(fname_csv,delimiter=',').astype(np.float32)
    pc_view(pc,CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT)
    
