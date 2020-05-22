#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/28 11:14
# @Author  : huangkai
# @Site    : 
# @File    : Stereo_BM.py
# @Software: PyCharm

import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
tic1=time.time()
imgL = cv2.imread('imag/teddy/im2.ppm',0)
imgR = cv2.imread('imag/teddy/im6.ppm',0)

# disparity range tuning
window_size = 5
min_disp = -1
max_disp = 79
num_disp = max_disp - min_disp

stereo = cv2.StereoBM_create(
    numDisparities=64,
    blockSize=11,
)

disparity = stereo.compute(imgL, imgR)

size1, size2 = disparity.shape

B = np.load("B.npz")
M = B['M1']
f = M[0,0]

depth=np.ones_like(disparity,dtype=np.uint8)
for i in range(size1):
    for j in range(size2):
        if abs(disparity[i][j])<5: ##噪音
            depth[i][j]=0
        else:
            depth[i][j]=f*100/disparity[i][j]
print(disparity)
print(depth)
# plt.subplot(221), plt.imshow(imgL)
# plt.subplot(222), plt.imshow(imgR)
plt.subplot(121), plt.imshow(disparity)
plt.subplot(122), plt.imshow(depth)
print('用时:',time.time()-tic1)
plt.show()

