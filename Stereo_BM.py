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

imgL = cv2.imread('imag/teddy/im2.ppm')
imgR = cv2.imread('imag/teddy/im6.ppm')

# disparity range tuning
window_size = 5
min_disp = -1
max_disp = 79
num_disp = max_disp - min_disp

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=3,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    disp12MaxDiff=1,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    # preFilterCap=63,
    # mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
disparity = stereo.compute(imgL, imgR).astype(np.float32)/ 16.0

size1, size2 = disparity.shape

B = np.load("B.npz")
M = B['M1']
f = M[0,0]
print(M,f,disparity)
depth=np.ones_like(disparity,dtype=np.uint8)
for i in range(size1):
    for j in range(size2):
        if abs(disparity[i][j])<5: ##噪音
            depth[i][j]=0
        else:
            depth[i][j]=f*100/disparity[i][j]

plt.subplot(221), plt.imshow(imgL)
plt.subplot(222), plt.imshow(imgR)
plt.subplot(223), plt.imshow(disparity)
plt.subplot(224), plt.imshow(depth)
plt.show()

