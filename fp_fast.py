#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 12:27
# @Author  : huangkai
# @Site    : 
# @File    : fp_fast.py
# @Software: PyCharm

import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
t = cv2.getTickCount()

img1 = cv2.imread('imag/box.png',0)          # queryImage
img2 = cv2.imread('imag/box_in_scene.png',0) # trainImage

# img1 = cv2.imread('imag/teddy/im2.ppm',0) # queryImage
# img2 = cv2.imread('imag/teddy/im6.ppm',0) # trainImage

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find the keypoints and descriptors with FAST
kp1 = fast.detect(img1,None)
kp2 = fast.detect(img2,None)

img1_kp = cv2.drawKeypoints(img1, kp1, None, (0,255,0), 0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, (0,255,0), 0)

t = (cv2.getTickCount()-t)/cv2.getTickFrequency()
print("time: ",t,"s")


print("kp1 = ", len(kp1))
print("kp2 = ", len(kp2))

cv2.imshow("img1", img1_kp)
cv2.imshow("img2", img2_kp)
cv2.waitKey()
cv2.destroyAllWindows()