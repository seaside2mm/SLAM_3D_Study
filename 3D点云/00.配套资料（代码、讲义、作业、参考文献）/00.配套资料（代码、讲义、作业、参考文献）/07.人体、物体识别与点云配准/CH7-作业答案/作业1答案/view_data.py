#!/usr/bin/python3
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pc_ds   =np.genfromtxt('ds.csv',delimiter=',')
ax=plt.figure().gca(projection='3d')
ax.plot(pc_ds[:,0],pc_ds[:,1],pc_ds[:,2],'.r',markersize=0.5)
ax.set_title('ds')
plt.show()

pc_scene=np.genfromtxt('scene.csv',delimiter=',')
ax=plt.figure().gca(projection='3d')
ax.plot(pc_scene[:,0],pc_scene[:,1],pc_scene[:,2],'.b',markersize=0.02)
ax.set_title('scene')
plt.show()

pc_out  =np.genfromtxt('out.csv',delimiter=',')
ax=plt.figure().gca(projection='3d')
ax.plot(pc_scene[:,0],pc_scene[:,1],pc_scene[:,2],'.b',markersize=0.5)
ax.plot(pc_out  [:,0],pc_out  [:,1],pc_out  [:,2],'.r',markersize=0.5)
ax.set_title('scene.csv with out.csv')
plt.show()
