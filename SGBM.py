#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 14:33
# @Author  : huangkai
# @Site    : 
# @File    : SGBM.py
# @Software: PyCharm
import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('imag/teddy/im2.ppm')
imgR = cv2.imread('imag/teddy/im6.ppm')

grayLeft = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayRight = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities=64,blockSize=11)
# stereo = cv2.StereoSGBM_create(numDisparities=64,blockSize=10)
# disparity = stereo.compute(imgL,imgR)
disparity = stereo.compute(grayLeft,grayRight)
plt.imshow(disparity,'gray')
plt.show()

# import numpy as np
# import cv2
#
#
# def update(val=0):
#     stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
#     stereo.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio', 'disparity'))
#     stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize', 'disparity'))
#     stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
#     stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'disparity'))
#
#     print('computing disparity...')
#     disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
#
#     cv2.imshow('left', imgL)
#     cv2.imshow('right', imgR)
#     cv2.imshow('disparity', (disp - min_disp) / num_disp)
#
#
# if __name__ == "__main__":
#     window_size = 5
#     min_disp = 16
#     num_disp = 192 - min_disp
#     blockSize = window_size
#     uniquenessRatio = 1
#     speckleRange = 12
#     speckleWindowSize = 3
#     disp12MaxDiff = 200
#     P1 = 600
#     P2 = 2400
#
#     imgL = cv2.imread('imag/teddy/im2.ppm')
#     imgR = cv2.imread('imag/teddy/im6.ppm')
#
#     cv2.namedWindow('disparity')
#     cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50, update)
#     cv2.createTrackbar('window_size', 'disparity', window_size, 21, update)
#     cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200, update)
#     cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, update)
#     cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, update)
#
#     stereo = cv2.StereoSGBM_create(
#         minDisparity=min_disp,
#         numDisparities=num_disp,
#         blockSize=window_size,
#         uniquenessRatio=uniquenessRatio,
#         speckleRange=speckleRange,
#         speckleWindowSize=speckleWindowSize,
#         disp12MaxDiff=disp12MaxDiff,
#         P1=P1,
#         P2=P2
#     )
#     update()
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
