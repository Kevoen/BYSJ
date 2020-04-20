#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 13:56
# @Author  : huangkai
# @Site    : 
# @File    : fp_brief.py
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

# # Initiate STAR detector
# star = cv2.FeatureDetector_create("STAR")

# FAST探测器
star = cv2.xfeatures2d.StarDetector_create()
# BRIEF提取器
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# 用STAR找到关键点
kp1 = star.detect(img1, None)
kp2 = star.detect(img2, None)

# 用BRIEF计算探测器的值
kp1, des1 = brief.compute(img1, kp1)
kp2, des2 = brief.compute(img2, kp2)

print("brief.descriptorSize: ",brief.descriptorSize())
print("des1.shape: ",des1.shape)
print("des2.shape: ",des2.shape)

img1_kp = cv2.drawKeypoints(img1, kp1, None, (0,255,0), 0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, (0,255,0), 0)

# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)
#
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
#
# # cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)


# FLANN parameters
FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# while using orb, the params is different with other
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)


t = (cv2.getTickCount()-t)/cv2.getTickFrequency()
print("time: ",t,"s")


print("kp1 = ",len(kp1))
print("kp2 = ",len(kp2))

cv2.imshow("brift", img3)
cv2.imshow("img1", img1_kp)
cv2.imshow("img2", img2_kp)
cv2.waitKey()
cv2.destroyAllWindows()

# plt.subplot(121),plt.imshow(img1_kp)
# plt.subplot(122),plt.imshow(img2_kp)
# plt.show()
# plt.imshow(img3),plt.show()