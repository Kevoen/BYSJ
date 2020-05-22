#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/9 8:44
# @Author  : huangkai
# @Site    : 
# @File    : sift_disparity.py
# @Software: PyCharm
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

MIN_MATCH_COUNT = 10
t = cv2.getTickCount()

# img1 = cv2.imread('imag/box.png',0)          # queryImage
# img2 = cv2.imread('imag/box_in_scene.png',0) # trainImage

img1 = cv2.imread('imag/teddy/im2.ppm',0) # queryImage
img2 = cv2.imread('imag/teddy/im6.ppm',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# draw key point

img1_kp = cv2.drawKeypoints(img1,kp1,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(img2,kp2,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

w,h = img1.shape[:2]
disparity = np.zeros((h,w))
for m in good:
    # print(kp1[m.queryIdx].pt[0])
    # print(kp2[m.trainIdx].pt[0])
    disp = (np.float32(kp1[m.queryIdx].pt[0]) - np.float32(kp2[m.trainIdx].pt[0]))/16.0
    l_x = np.int(kp1[m.queryIdx].pt[0])
    l_y = np.int(kp1[m.queryIdx].pt[1])
    disparity[l_x][l_y] = disp
    print(disp)
print(disparity)
plt.imshow(disparity),plt.show()



    # disp = np.array([kp1[m.queryIdx].pt[0],)
    # disp = [m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]
    # disparity = np.array([kp1[m.queryIdx],disp])

# if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()
#
#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
#
#     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#
#     # warped_image = cv2.warpPerspective(img1, M, (img1.shape[1]+img2.shape[1], img2.shape[0]))
#     # cv2.imshow('warped_image',warped_image)
# else:
#     print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
#     matchesMask = None

# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
#                     #flags = 0 时显示singlePoints
#
#
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#
# t = (cv2.getTickCount()-t)/cv2.getTickFrequency()