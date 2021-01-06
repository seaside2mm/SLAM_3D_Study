import matplotlib.pyplot as plt 
import matplotlib.tri as mtri
import numpy as np
import cv2


np.random.seed(124)
N = 100
idx = list(range(N))
xs = np.random.randint(1,98,N).tolist()
ys = np.random.randint(1,98,N).tolist()

plt.scatter(xs, ys)
plt.show()

subdiv = cv2.Subdiv2D((0,0,100,100))
for x,y in zip(xs,ys):
    subdiv.insert((x,y))
tri_cv2 = subdiv.getTriangleList()

XS, YS, IS = [],[],[]
for n in range(len(tri_cv2)):
    t = tri_cv2[n]   #t包含了三角形顶点坐标
    XS.extend([t[0],t[2],t[4]])   #3个x坐标
    YS.extend([t[1],t[3],t[5]])   #3个y坐标
    IS.append([n*3, n*3+1, n*3+2]) #面对应顶点序号
    
fig, ax = plt.subplots()
ax.triplot(mtri.Triangulation(XS,YS,IS),'k-')
plt.show()