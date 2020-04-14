#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 14:25
# @Author  : huangkai
# @Site    : 
# @File    : KAZE
# @Software: PyCharm

import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
t = cv2.getTickCount()

# img1 = cv2.imread('imag/box.png',0)          # queryImage
# img2 = cv2.imread('imag/box_in_scene.png',0) # trainImage

img1 = cv2.imread('imag/teddy/im2.ppm',0) # queryImage
img2 = cv2.imread('imag/teddy/im6.ppm',0) # trainImage

# Initiate KAZE detector
# kaze = cv2.KAZE_create()
kaze = cv2.AKAZE_create()

# find the keypoints and descriptors with BRISK
kp1, des1 = kaze.detectAndCompute(img1,None)
kp2, des2 = kaze.detectAndCompute(img2,None)

img1_kp = cv2.drawKeypoints(img1, kp1, None, (0,255,0), 4)
img2_kp = cv2.drawKeypoints(img2, kp2, None, (0,255,0), 4)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)


# # FLANN parameters
# FLANN_INDEX_KDTREE = 0
# # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# # while using orb, the params is different with other
# FLANN_INDEX_LSH = 6
# index_params= dict(algorithm = FLANN_INDEX_LSH,
#                    table_number = 6, # 12
#                    key_size = 12,     # 20
#                    multi_probe_level = 1) #2
#
# search_params = dict(checks=50)   # or pass empty dictionary
#
# flann = cv2.FlannBasedMatcher(index_params,search_params)
#
# matches = flann.knnMatch(des1,des2,k=2)
#
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
#
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
#
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 0)
#
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)


t = (cv2.getTickCount()-t)/cv2.getTickFrequency()
print("time: ",t,"s")


print("kp1 = ",len(kp1))
print("kp2 = ",len(kp2))

cv2.imshow("surf", img3)
cv2.imshow("img1", img1_kp)
cv2.imshow("img2", img2_kp)
cv2.waitKey()
cv2.destroyAllWindows()