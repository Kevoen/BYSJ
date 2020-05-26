#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 9:48
# @Author  : huangkai
# @Site    : 
# @File    : SAD.py
# @Software: PyCharm


import os
import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from PIL import Image
maxDisparity=25 #最大视差
window_size=13#滑动窗口大小

lraw=cv.imread('imag/teddy/im2.ppm')
rraw=cv.imread('imag/teddy/im6.ppm')

limg= cv.cvtColor(lraw,cv.COLOR_BGR2GRAY)
rimg= cv.cvtColor(rraw,cv.COLOR_BGR2GRAY)
limg=np.asanyarray(limg,dtype=np.double)
rimg=np.asanyarray(rimg,dtype=np.double)
img_size=np.shape(limg)[0:2]
# plt.imshow(limg)
# plt.show()
# plt.imshow(rimg)
# plt.show()

tic1=time.time()
imgDiff=np.zeros((img_size[0],img_size[1],maxDisparity))
e = np.zeros(img_size)
for i in range(0,maxDisparity):
    e=np.abs(rimg[:,0:(img_size[1]-i)]- limg[:,i:img_size[1]])
    e2=np.zeros(img_size) #计算窗口内的和
    for x in range((window_size),(img_size[0]-window_size)):
        for y in range((window_size),(img_size[1]-window_size)):
            e2[x,y]=np.sum(e[(x-window_size):(x+window_size),(y-window_size):(y+window_size)])
        imgDiff[:,:,i]=e2
dispMap=np.zeros(img_size)

for x in range(0,img_size[0]):
    for y in range(0,img_size[1]):
        val=np.sort(imgDiff[x,y,:])
        if np.abs(val[0]-val[1])>10:
            val_id=np.argsort(imgDiff[x,y,:])
            dispMap[x,y]=val_id[0]/maxDisparity*255
# Show disparity map before generating 3D cloud to verify that point cloud will be usable.
size1, size2 = dispMap.shape

B = np.load("B.npz")
M = B['M1']
f = M[0,0]
print(M,f,dispMap)
depth=np.ones_like(dispMap,dtype=np.uint8)
for i in range(size1):
    for j in range(size2):
        if abs(dispMap[i][j])<5: ##噪音
            depth[i][j]=0
        else:
            depth[i][j]=f*65/dispMap[i][j]

# print(depth)

print('用时:',time.time()-tic1)
# plt.subplot(221), plt.imshow(limg)
# plt.subplot(222), plt.imshow(rimg)
plt.subplot(121), plt.imshow(dispMap)
plt.subplot(122), plt.imshow(depth)
plt.show()