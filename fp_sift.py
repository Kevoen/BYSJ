#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/7 10:42
# @Author  : huangkai
# @Site    : 
# @File    : fp_sift.py
# @Software: PyCharm

# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# MIN_MATCH_COUNT = 10
# t = cv2.getTickCount()
#
# img1 = cv2.imread('imag/box.png',0)          # queryImage
# img2 = cv2.imread('imag/box_in_scene.png',0) # trainImage
#
# # img1 = cv2.imread('imag/teddy/im2.ppm',0) # queryImage
# # img2 = cv2.imread('imag/teddy/im6.ppm',0) # trainImage
#
# # Initiate SIFT detector
# sift = cv2.xfeatures2d.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
#
# # draw key point
# img1_kp = cv2.drawKeypoints(img1,kp1,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img2_kp = cv2.drawKeypoints(img2,kp2,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # FLANN parameters
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)
#
# flann = cv2.FlannBasedMatcher(index_params,search_params)
#
# matches = flann.knnMatch(des1,des2,k=2)
#
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
# num = 0
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
#         num = num + 1
#
#
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = 0)
#
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
#
# t = (cv2.getTickCount()-t)/cv2.getTickFrequency()
# print("time: ", t, 's')
#
# print("img1's total founded keypoints :", len(kp1))
# print("img2's total founded keypoints :", len(kp2))
# print("total matching points :", len(matches))
# print("final matching points :", num)
# print("precision :", num/len(matches))
# print("recall :", num/len(kp1))
#
# # plt.subplot(121),plt.imshow(img1_kp)
# # plt.subplot(122),plt.imshow(img2_kp)
# # plt.show()
# # plt.imshow(img3),plt.show()
#
# cv2.imshow('img1',img1_kp)
# cv2.imshow('img2',img2_kp)
# cv2.imshow('sift',img3)
# cv2.waitKey()
# cv2.destroyAllWindows()


#***************改进SIFT特征点匹配（RANSAC）*****************#

import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
t = cv2.getTickCount()

# img1 = cv2.imread('imag/box.png',0)          # queryImage
# img2 = cv2.imread('imag/box_in_scene.png',0) # trainImage

# img1 = cv2.imread('imag/mychessborad/fan01.jpg',0) # queryImage
# img2 = cv2.imread('imag/mychessborad/fan02.jpg',0) # trainImage

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

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # 变换
    warped_image = cv2.warpPerspective(img1, M, (img1.shape[1]+img2.shape[1], img2.shape[0]))
    cv2.imshow('warped_image',warped_image)
else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
                    #flags = 0 时显示singlePoints


img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

t = (cv2.getTickCount()-t)/cv2.getTickFrequency()
print("time: ",t,"s")

print("img1's total founded keypoints :", len(kp1))
print("img2's total founded keypoints :", len(kp2))
print("total matching points :", len(matches))
print("final matching points :", len(matchesMask))
print("precision :", len(matchesMask)/len(matches))
print("recall :",len(matchesMask)/len(kp1))

# plt.imshow(img1_kp),plt.show()
# plt.imshow(img2_kp),plt.show()
# plt.imshow(img3),plt.show()

cv2.imshow('sift',img3)
cv2.imshow('img1',img1_kp)
cv2.imshow('img2',img2_kp)
cv2.waitKey()
cv2.destroyAllWindows()