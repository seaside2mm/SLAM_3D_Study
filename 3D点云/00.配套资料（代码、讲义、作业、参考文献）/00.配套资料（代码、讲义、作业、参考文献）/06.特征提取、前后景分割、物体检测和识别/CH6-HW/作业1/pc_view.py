
# 查看3D点云
import numpy as np
import cv2
import time

def pc_trans_movz(tz): return pc_trans_mov(0,0,tz)

def pc_trans_rotx(b):    
    return np.array([[1,        0 ,       0 ,0],
                    [0, np.cos(b),np.sin(b),0],
                    [0,-np.sin(b),np.cos(b),0],
                    [0,        0 ,       0 ,1]])


def pc_trans_roty(b):    
    return np.array([[np.cos(b),0,-np.sin(b),0],
                    [       0 ,1,        0 ,0],
                    [np.sin(b),0, np.cos(b),0],
                    [       0 ,0,        0 ,1]])

def pc_trans_mov(tx,ty,tz):    
    return np.array([[ 1, 0, 0,0],
                    [ 0, 1, 0,0],
                    [ 0, 0, 1,0],
                    [tx,ty,tz,1]])

def pc_view(pc,\
            cam_fx,cam_fy,cam_cx,cam_cy,\
            img_wid,img_hgt,\
            cz=None,dmin=None,dmax=None,name='point colud viewer'):
    
    cv2.namedWindow(name)
    
    eps=1.0e-8
    
    if dmin is None: dmin=np.min(pc[:,2])   # 伪彩色距离范围
    if dmax is None: dmax=np.max(pc[:,2])
        
    # 点云变换参数
    ax=ay=0.0                       # 点云旋转角度
    mz=0                            # 点云观察点位置
    if cz is None: 
        cz=np.mean(pc[:,2])         # 点云旋转中心
    M=np.eye(4,dtype=np.float32)    # 点云变换矩阵
    update_M=False                  # 标示变换矩阵是否更新了
    
    print('dmin:',dmin)
    print('dmax:',dmin)
    print('cz:',cz)


    # 鼠标动作跟踪
    mouse_x=mouse_y=0
    mouse_down=False
    
    def mouse_callback(evt,x,y,flags,param):
        nonlocal mouse_x,mouse_y,mouse_down
        nonlocal ax,ay,mz,update_M
        
        if evt==cv2.EVENT_LBUTTONDOWN:
            mouse_down=True
            mouse_x,mouse_y=x,y
        elif evt==cv2.EVENT_LBUTTONUP:
            mouse_down=False
        elif evt==cv2.EVENT_MOUSEMOVE:
            if mouse_down:
                if flags&cv2.EVENT_FLAG_SHIFTKEY==0:
                    dx,dy=x-mouse_x,y-mouse_y
                    mouse_x,mouse_y=x,y
                    if not update_M:
                        ax+=dy/50.0
                        ay-=dx/50.0
                        update_M=True
                else:
                    dy=y-mouse_y
                    mouse_x,mouse_y=x,y
                    if not update_M:
                        mz+=dy/10
                        update_M=True
                
    cv2.setMouseCallback(name,mouse_callback)
    
    pc_view=pc.copy()
    while True:            
        # 点云变换，并将变换后的点云映射回深度图
        if update_M:
            M=np.dot(pc_trans_movz(-cz),pc_trans_rotx(ax))
            M=np.dot(M,pc_trans_roty(ay))
            M=np.dot(M,pc_trans_movz(cz+mz))
            
            pc_view=np.dot(pc,M[:3,:3])+M[3,:3]
            update_M=False
        
        # 去除镜头后方的点
        valid=pc_view[:,2]>eps
        z=pc_view[valid,2]
        
        # 点云反向映射到像素坐标位置
        u=np.round(pc_view[valid,0]*cam_fx/z+cam_cx).astype(int)
        v=np.round(pc_view[valid,1]*cam_fy/z+cam_cy).astype(int)
    
        # 滤除超出图像尺寸的无效像素
        valid=np.bitwise_and(np.bitwise_and((u>=0),(u<img_wid)),
                             np.bitwise_and((v>=0),(v<img_hgt)))
        u,v,z=u[valid],v[valid],z[valid]
    
        # 按距离填充生成深度图，近距离覆盖远距离
        img_z=np.full((img_hgt, img_wid),np.inf)        
        for ui,vi,zi in zip(u,v,z):
            img_z[vi,ui]=min(img_z[vi,ui],zi)   # 近距离像素屏蔽远距离像素
        mask=np.isinf(img_z)                    # mask标示出未填充的深度图像素
        
        # 将深度图转换成伪彩色，并更新显示
        img_u8=np.uint8(np.clip((img_z-dmin)/float(dmax-dmin),0.0,1.0)*255)
        img_rgb=cv2.applyColorMap(255-img_u8,cv2.COLORMAP_RAINBOW)
        img_rgb[mask,:]=0
        
        # 刷新屏幕显示
        cv2.imshow(name, img_rgb)
        
        # 检查用户界面操作
        key=cv2.waitKey(1)&0xFF
        if key==ord('q') or key==27 or cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE) < 1:
            break

        time.sleep(0.001)
    return


class pc_view_c:
    def __init__(self,
                 cam_fx,cam_fy,cam_cx,cam_cy,\
                 img_wid,img_hgt,\
                 cz=1,dmin=0,dmax=2,name='point colud viewer',eps=1.0e-8):
        
        cv2.namedWindow(name)
        
        # 相机参数
        self.fx,self.fy=cam_fx,cam_fy
        self.cx,self.cy=cam_cx,cam_cy
        self.w,self.h=img_wid,img_hgt
        
        # 显示参数
        self.cz=cz
        self.dmin,self.dmax=dmin,dmax
        self.name=name
        
        # 用于深度图变换的LUT
        x,y=np.meshgrid(range(self.w),range(self.h))
        x=x.astype(np.float32)-cam_cx
        y=y.astype(np.float32)-cam_cy
        f=(cam_fx+cam_fy)*0.5
        self.cos_theta=f/np.sqrt(x**2+y**2+f**2)
        self.tab_x=x/cam_fx 
        self.tab_y=y/cam_fy
        
        self.eps=eps

        self.pc=None
        
        # 点云变换参数
        self.ax=self.ay=0.0                 # 点云旋转角度
        self.mz=0                           # 点云观察点位置
        self.M=np.eye(4,dtype=np.float32)   # 点云变换矩阵
        self.update_M=False                 # 标示变换矩阵是否更新了
        
        # 鼠标动作跟踪
        self.mouse_x=self.mouse_y=0
        self.mouse_down=False
                    
        cv2.setMouseCallback(self.name,self.mouse_callback)

    def img_z_to_pc(self,img_z):
        pc_x=img_z*self.tab_x    #   X=Z*(u-cx)/fx
        pc_y=img_z*self.tab_y    #   Y=Z*(v-cy)/fy
        pc_z=img_z
        return np.array([pc_x.ravel(),pc_y.ravel(),pc_z.ravel()]).T
    
    def dep_to_z(self,img): return img*self.cos_theta
    def z_to_dep(self,img_z): return img_z/self.cos_theta

    def dep_to_pc(self,img):
        pc_z=img*self.cos_theta
        pc_x=pc_z*self.tab_x    #   X=Z*(u-cx)/fx
        pc_y=pc_z*self.tab_y    #   Y=Z*(v-cy)/fy
        return np.array([pc_x.ravel(),pc_y.ravel(),pc_z.ravel()]).T

    def pc_to_img_z(self,pc):
        z=pc[:,2]
        z[np.abs(z)<eps]=eps    # 防止后面的除零错

        # 反向映射到像素坐标位置
        u=np.round(pc[:,0]*self.fx/z+self.cx).astype(int)
        v=np.round(pc[:,1]*self.fy/z+self.cy).astype(int)
        
        # 滤除超出图像尺寸的无效像素
        valid=np.bitwise_and(np.bitwise_and((u>=0),(u<self.w)),
                             np.bitwise_and((v>=0),(v<self.h)))
        u_valid=u[valid]
        v_valid=v[valid]
        z_valid=z[valid]
        
        # 按距离填充生成深度图，近距离覆盖远距离
        img_z=np.full((hgt, wid),np.inf)        
        for ui,vi,zi in zip(u_valid,v_valid,z_valid):
            img_z[vi,ui]=min(img_z[vi,ui],zi)   # 近距离像素屏蔽远距离像素
        return img_z

    def mouse_callback(self,evt,x,y,flags,param):        
        if evt==cv2.EVENT_LBUTTONDOWN:
            self.mouse_down=True
            self.mouse_x,self.mouse_y=x,y
        elif evt==cv2.EVENT_LBUTTONUP:
            self.mouse_down=False
        elif evt==cv2.EVENT_MOUSEMOVE:
            if self.mouse_down:
                if flags&cv2.EVENT_FLAG_SHIFTKEY==0:
                    dx,dy=x-self.mouse_x,y-self.mouse_y
                    self.mouse_x,self.mouse_y=x,y
                    if not self.update_M:
                        self.ax+=dy/50.0
                        self.ay-=dx/50.0
                        self.update_M=True
                else:
                    dy=y-self.mouse_y
                    self.mouse_x,self.mouse_y=x,y
                    if not self.update_M:
                        self.mz+=dy/10
                        self.update_M=True

    def update_img_dep(self,img):
        self.update_pc(self.dep_to_pc(img))

    def update_pc(self,pc=None):
        if pc is not None:
            self.pc=pc.copy()
            self.pc_view=np.dot(pc,self.M[:3,:3])+self.M[3,:3]


    def update_trans_mat(self):
        # 点云变换，并将变换后的点云映射回深度图
        if self.update_M:
            
            self.M=np.dot(pc_trans_movz(-self.cz),pc_trans_rotx(self.ax))
            self.M=np.dot(self.M,pc_trans_roty(self.ay))
            self.M=np.dot(self.M,pc_trans_movz(self.cz+self.mz))
            
            self.pc_view=np.dot(self.pc,self.M[:3,:3])+self.M[3,:3]
            self.update_M=False


    def pc_view_to_img_rgb(self,pc):
        # 去除镜头后方的点
        valid=pc[:,2]>self.eps
        z=pc[valid,2]
        
        # 点云反向映射到像素坐标位置
        u=np.round(pc[valid,0]*self.fx/z+self.cx).astype(int)
        v=np.round(pc[valid,1]*self.fy/z+self.cy).astype(int)
    
        # 滤除超出图像尺寸的无效像素
        valid=np.bitwise_and(np.bitwise_and((u>=0),(u<self.w)),
                             np.bitwise_and((v>=0),(v<self.h)))
        u,v,z=u[valid],v[valid],z[valid]
    
        # 按距离填充生成深度图，近距离覆盖远距离
        img_z=np.full((self.h, self.w),np.inf)        
        for ui,vi,zi in zip(u,v,z):
            img_z[vi,ui]=min(img_z[vi,ui],zi)   # 近距离像素屏蔽远距离像素
        mask=np.isinf(img_z)                    # mask标示出未填充的深度图像素
        
        # 将深度图转换成伪彩色，并更新显示
        img_u8=np.uint8(np.clip((img_z-self.dmin)/float(self.dmax-self.dmin),0.0,1.0)*255)
        img_rgb=cv2.applyColorMap(255-img_u8,cv2.COLORMAP_RAINBOW)
        img_rgb[mask,:]=0
        return img_rgb


    def update_view(self,img):
        return self.update_pc_view(self.dep_to_pc(img))
        
    def update_pc_view(self,pc=None):
        update = self.update_M or pc is not None
        
        self.update_pc(pc)
        self.update_trans_mat()

        # 刷新屏幕显示
        if update:
            img_rgb=self.pc_view_to_img_rgb(self.pc_view)
            cv2.imshow(self.name, img_rgb)
        
        # 检查用户界面操作
        key=cv2.waitKey(1)&0xFF
        return False if key==ord('q') or \
                        key==27       or \
                        cv2.getWindowProperty(self.name,cv2.WND_PROP_VISIBLE)<1\
                     else True
    
##########
# 用法演示
##########
if __name__=='__main__':
    from pc_to_dep import *
    
    # 相机参数
    CAM_WID,CAM_HGT = 320,240
    CAM_FX,CAM_FY   = 200,200
    CAM_CX,CAM_CY   = CAM_WID//2,CAM_HGT//2
    IMG_HGT,IMG_WID = CAM_HGT,CAM_WID

    tab_x=tab_y=cos_theta=None
    frames=np.load('cube_dep.npy')

    if False:
        N=50000
        pc=np.random.rand(N,3)-0.5
        pc=pc/np.linalg.norm(pc,axis=1).reshape(N,1)
        pc=pc*np.array([0.5,0.3,0.4])+[0,0,0.8]
        pc_view(pc,CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT)
    else:
        viewer=pc_view_c(CAM_FX,CAM_FY,CAM_CX,CAM_CY,CAM_WID,CAM_HGT)
        cnt=0
        for img_dep in frames:
            if not viewer.update_view(img_dep): break
            print(cnt)
            cnt+=1
            
