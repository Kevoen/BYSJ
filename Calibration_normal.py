#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 10:08
# @Author  : huangkai
# @Site    : 
# @File    : Calibration_normal.py
# @Software: PyCharm

import numpy as np
import cv2
import glob
import os


def calib_normal(inter_corner_shape, img_dir, img_type):
    # termination criteria 阈值
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    w, h = inter_corner_shape

    objp = np.zeros((w*h, 3),np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    # 存储棋盘格的世界坐标和图像坐标
    objpoints = []      #世界空间的三维点
    imgpoints = []      #图像平面的二维点

    images = glob.glob(img_dir + os.sep + 'left/**.' + img_type)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #寻找棋盘角点
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

        #如果查找到则保存世界坐标和图像坐标，并再次查找下一张
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

        #绘制显示角点
        img = cv2.drawChessboardCorners(img, (w, h), corners2, ret)
        cv2.imshow('FoundCorners', img)
        cv2.waitKey()

    cv2.destroyAllWindows()

#相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(("ret:"), ret)
    print(("internal matrix:\n"), mtx)
    # in the form of (k_1,k_2,p_1,p_2,k_3)
    print(("distortion cofficients:\n"), dist)
    #R,T
    print(("rotation vectors:\n"), rvecs)
    print(("translation vectors:\n"), tvecs)


if __name__ == '__main__':
    inter_corner_shape = (7, 6)
    img_dir = "imag/chessboard"
    img_type = "jpg"
    calib_normal(inter_corner_shape, img_dir, img_type)