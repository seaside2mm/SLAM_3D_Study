#!/usr/bin/python3
# coding=utf-8

import glob,time
import numpy as np
import matplotlib.pyplot as plt

for fname_csv in glob.glob('*.csv'):
    img_csv=np.genfromtxt(fname_csv,delimiter=',').astype(np.float32)
    plt.clf()
    plt.imshow(img_csv,cmap='jet')
    plt.title(fname_csv)
    plt.show(block=False)
    plt.pause(0.0001)
