#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 9:29
# @Author  : huangkai
# @Site    : 
# @File    : StereoCalibration.py
# @Software: PyCharm

import numpy as np
import cv2 as cv
import glob
import os
from matplotlib import pyplot as plt

def stereo_cal(inter_corner_shape, img_dir, img_type):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_cal = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 30, 1e-5)

    w, h = inter_corner_shape

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((h * w, 3), np.float32)
    objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    # images = glob.glob('imag/chessboard/*.jpg')
    images_left = glob.glob(img_dir + os.sep + 'left/**.' + img_type)
    images_right = glob.glob(img_dir + os.sep + 'right/**.' + img_type)
    images_left.sort()
    images_right.sort()

    for i, fname in enumerate(images_right):
        print(str(i + 1) + " out of " + str(len(images_right)))
        img_l = cv.imread(images_left[i])
        img_r = cv.imread(images_right[i])

        gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_l, corners_l = cv.findChessboardCorners(gray_l, (w, h), None)
        ret_r, corners_r = cv.findChessboardCorners(gray_r, (w, h), None)

        # If found, add object points, image points (after refining them)


        if ret_l == True:
            print("image " + str(i + 1) + "left - io")
            corners2_l = cv.cornerSubPix(gray_l,corners_l, (11,11), (-1,-1), criteria)
        else:
            print("image "+ str(i + 1) +"ret_l is False")

        if ret_r == True:
            print("image " + str(i + 1) + "right - io")
            corners2_r = cv.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        else:
            print("image "+ str(i + 1) +"ret_r is False")

        if (ret_l == True) & (ret_r == True):
            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)

            # Draw and display the corners
            plt.subplot(121)
            cv.drawChessboardCorners(img_l, (w, h), corners2_l, ret_l)
            plt.imshow(img_l)
            plt.subplot(122)
            cv.drawChessboardCorners(img_r, (w, h), corners2_r, ret_r)
            plt.imshow(img_r)
            plt.show()

    ### calibration###
    #calicrate left camera
    ret, M1, d1, r1, t1 = cv.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1], None, None)

    # calicrate right camera
    ret, M2, d2, r2, t2 = cv.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1], None, None)

    # stereo calibration
    flags = (cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6)

    stereocalib_criteria = (cv.TERM_CRITERIA_MAX_ITER +cv.TERM_CRITERIA_EPS, 100, 1e-5)

    flags = (cv.CALIB_FIX_PRINCIPAL_POINT | cv.CALIB_FIX_ASPECT_RATIO | cv.CALIB_FIX_FOCAL_LENGTH |
             cv.CALIB_FIX_INTRINSIC | cv.CALIB_FIX_K3 | cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 |
             cv.CALIB_FIX_K6)

    T = np.zeros((3, 1), dtype=np.float64)
    R = np.eye(3, dtype=np.float64)

    ret, M1, d1, M2, d2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_l,
        imgpoints_r, M1, d1, M2,
        d2, gray_l.shape[::-1],
        criteria=stereocalib_criteria,
        flags=flags)

    # get new optimal camera matrix
    newCamMtx1, roi1 = cv.getOptimalNewCameraMatrix(M1, d1, gray_l.shape[::-1], 0, gray_l.shape[::-1])
    newCamMtx2, roi2 = cv.getOptimalNewCameraMatrix(M2, d2, gray_l.shape[::-1], 0, gray_l.shape[::-1])

    np.savez('B.npz', M1=M1, d1=d1, M2=M2, d2=d2, R=R, T=T, newCamMtx1=newCamMtx1, newCamMtx2=newCamMtx2)

    # # calculate the error of reproject
    # mean_error = 0
    # for i in range(len(objpoints)):
    #     imgpoints2, _ = cv.projectPoints(objpoints[i], R[i], T[i], newCamMtx1, d1)
    #     error = cv.norm(imgpoints_l[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    #     mean_error += error
    # print("left:")
    # print( "total error: {}".format(mean_error/len(objpoints)) )
    #
    # for i in range(len(objpoints)):
    #     imgpoints2, _ = cv.projectPoints(objpoints[i], R[i], T[i], newCamMtx2, d2)
    #     error = cv.norm(imgpoints_r[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    #     mean_error += error
    # print("right:")
    # print( "total error: {}".format(mean_error/len(objpoints)) )

    return newCamMtx1, newCamMtx2, d1, d2, R, T, E, F

if __name__ == '__main__':
    inter_corner_shape = (7, 6)
    img_dir = "imag/chessboard"
    img_type = "jpg"
    stereo_cal(inter_corner_shape, img_dir, img_type)