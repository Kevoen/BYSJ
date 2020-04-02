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

            # # Draw and display the corners
            # plt.subplot(121)
            # cv.drawChessboardCorners(img_l, (w, h), corners2_l, ret_l)
            # plt.imshow(img_l)
            # plt.subplot(122)
            # cv.drawChessboardCorners(img_r, (w, h), corners2_r, ret_r)
            # plt.imshow(img_r)
            # plt.show()

    # get shape
    img_shape = gray_l.shape[::-1]

    ### calibration###
    #calicrate left camera
    ret, M1, d1, r1, t1 = cv.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)

    # calicrate right camera
    ret, M2, d2, r2, t2 = cv.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)

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
        d2, img_shape,
        criteria=stereocalib_criteria,
        flags=flags)


    np.savez('B.npz', M1=M1, d1=d1, M2=M2, d2=d2, R=R, T=T, F=F, E=E, img_shape=img_shape)

    ## calculate the error of reproject
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], r1[i], t1[i], M1, d1)
        error = cv.norm(imgpoints_l[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("left:")
    print( "total error: {}".format(mean_error/len(objpoints)) )

    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], r2[i], t2[i], M2, d2)
        error = cv.norm(imgpoints_r[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("right:")
    print( "total error: {}".format(mean_error/len(objpoints)) )

    # return img_shape


def stereo_rect(inter_corner_shape, img_dir, img_type):

    #load M,dist,R,T,E
    B = np.load('B.npz')
    M1 = B['M1']
    M2 = B['M2']
    d1 = B['d1']
    d2 = B['d2']
    R = B['R']
    T = B['T']
    E = B['E']
    F = B['F']

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w, h = inter_corner_shape

    # images_left = glob.glob(img_dir + os.sep + 'left01.' + img_type)
    # images_right = glob.glob(img_dir + os.sep + 'right01.' + img_type)

    img_l = cv.imread("imag/chessboard/left01.jpg")
    img_r = cv.imread("imag/chessboard/right01.jpg")

    # convert to cv2
    img_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
    img_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

    img_shape = img_l.shape[::-1]

    ##rectiftication and undistrotion##
    # get new optimal camera matrix
    newCamMtx1, roi1 = cv.getOptimalNewCameraMatrix(M1, d1, img_shape, 0, img_shape)
    newCamMtx2, roi2 = cv.getOptimalNewCameraMatrix(M2, d2, img_shape, 0, img_shape)

    #Computes rectification transforms for each head of a calibrated stereo camera.
    (rectification_l, rectification_r, projection_l,projection_r, disparityToDepthMap, ROI_l, ROI_r) = cv.stereoRectify(
        M1, d1, M2, d2, img_shape, R, T,None, None, None, None, None,alpha=0)

    #Computes the undistortion and rectification transformation map.
    # leftMapX, leftMapY = cv.initUndistortRectifyMap(M1, d1, rectification_l,projection_l,img_shape, cv.CV_32FC1)
    # rightMapX, rightMapY = cv.initUndistortRectifyMap(M2, d2, rectification_r, projection_r,img_shape, cv.CV_32FC1)

    leftMapX, leftMapY = cv.initUndistortRectifyMap(M1, d1, None, newCamMtx1, img_shape, cv.CV_32FC1)
    rightMapX, rightMapY = cv.initUndistortRectifyMap(M2, d2, None, newCamMtx2, img_shape, cv.CV_32FC1)

    #Applies a generic geometrical transformation to an image.
    imglCalRect = cv.remap(img_l, leftMapX, leftMapY, cv.INTER_LINEAR)
    imgrCalRect = cv.remap(img_r, rightMapX, rightMapY, cv.INTER_LINEAR)

    # x1, y1, w1, h1 = roi1
    # x2, y2, w2, h2 = roi2
    # imglCalRect = imglCalRect[y1:y1 + h1, x1:x1 + w1]
    # imgrCalRect = imgrCalRect[y2:y2 + h2, x2:x2 + w2]
    cv.imshow("calibRect_l", imglCalRect)
    cv.imshow("calibRect_r", imgrCalRect)
    cv.imshow("img",img_l)
    cv.waitKey()
    cv.destroyAllWindows()

    # numpyHorizontalCalibRect = np.hstack((imglCalRect, imgrCalRect))
    # cv.imshow("calibRect", numpyHorizontalCalibRect)
    # cv.waitKey(500)
    # cv.destroyAllWindows()


if __name__ == '__main__':
    inter_corner_shape = (7, 6)
    img_dir = "imag/chessboard"
    img_type = "jpg"
    # stereo_cal(inter_corner_shape, img_dir, img_type)
    stereo_rect(inter_corner_shape, img_dir, img_type)