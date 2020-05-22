#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/28 15:02
# @Author  : huangkai
# @Site    : 
# @File    : epipolar_rectify.py
# @Software: PyCharm

# imports
import numpy as np
import cv2
import glob
import argparse
import sys
import os

# size calib array
numEdgeX = 7
numEdgeY = 6

try:
    # define pair
    # p = 1
    # cal_path = pathCalib + "\\pair" + str(p)
    cal_path = "imag/chessboard"

    images_right = glob.glob(cal_path + os.sep + 'right/*.jpg')
    images_left = glob.glob(cal_path + os.sep + 'left/*.jpg')
    images_left.sort()
    images_right.sort()

    # termination criteria
    criteria = (cv2.TermCriteria_EPS +
                    cv2.TermCriteria_MAX_ITER, 30, 0.001)
    criteria_cal = (cv2.TermCriteria_EPS +
                    cv2.TermCriteria_MAX_ITER, 30, 1e-5)

    # prepare object points, like (0,0,0); (1,0,0); ...; (6,5,0)
    objp = np.zeros((numEdgeX*numEdgeY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:numEdgeX, 0:numEdgeY].T.reshape(-1, 2)

    objpoints = []     # 3d points in real world space
    imgpoints_l = []   # 2d points in image plane for calibration
    imgpoints_r = []   # 2d points in image plane for calibration

    for i, fname in enumerate(images_right):
        print(str(i+1) + " out of " + str(len(images_right)))
        img_l = cv2.imread(images_left[i])
        img_r = cv2.imread(images_right[i])

        # convert to cv2
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # find the chess board corners
        ret_l, corners_l = cv2.findChessboardCorners(img_l, (numEdgeX, numEdgeY), None)
        ret_r, corners_r = cv2.findChessboardCorners(img_r, (numEdgeX, numEdgeY), None)

        # objpoints.append(objp)

        if ret_l is True:
            print("image " + str(i+1) + "left - io")
            rt = cv2.cornerSubPix(img_l, corners_l, (11, 11),
                                  (-1, -1), criteria)
            # imgpoints_l.append(corners_l)

        if ret_r is True:
            print("image " + str(i+1) + "right - io")
            rt = cv2.cornerSubPix(img_r, corners_r, (11, 11),
                                  (-1, -1), criteria)
            # imgpoints_r.append(corners_r)

        if (ret_l == True) & (ret_r == True):
            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)

        # get shape
        img_shape = img_l.shape[::-1]


    ### CALIBRATION ###
    # calibrate left camera
    rt, M1, d1, r1, t1 = cv2.calibrateCamera(
    objpoints, imgpoints_l, img_shape, None, None)

    # calibrate right camera
    rt, M2, d2, r2, t2 = cv2.calibrateCamera(
    objpoints, imgpoints_r, img_shape, None, None)

    # stereo calibration
    flags = (cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6)

    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                    cv2.TERM_CRITERIA_EPS, 100, 1e-5)


    flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_FOCAL_LENGTH |
         cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 |
         cv2.CALIB_FIX_K6)

    T = np.zeros((3, 1), dtype=np.float64)
    R = np.eye(3, dtype=np.float64)

    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l,
        imgpoints_r, M1, d1, M2,
        d2, img_shape,
        criteria = stereocalib_criteria,
        flags=flags)

    # get new optimal camera matrix
    newCamMtx1, roi1 = cv2.getOptimalNewCameraMatrix(M1, d1, img_shape, 0, img_shape)
    newCamMtx2, roi2 = cv2.getOptimalNewCameraMatrix(M2, d2, img_shape, 0, img_shape)


    # rectification and undistortion maps which can be used directly to correct the stereo pair
    (rectification_l, rectification_r, projection_l,
        projection_r, disparityToDepthMap, ROI_l, ROI_r) = cv2.stereoRectify(
            M1, d1, M2, d2, img_shape, R, T,
            None, None, None, None, None,
            alpha=0)
            #cv2.CALIB_ZERO_DISPARITY,
            #  principal points of each camera have the same pixel coordinates in rect views
            # alpha=1 no pixels lost, alpha=0 pixels lost

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        M1, d1, rectification_l, projection_l,
        img_shape, cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        M2, d2, rectification_r, projection_r,
        img_shape, cv2.CV_32FC1)



    ### UNCALIBRATED RECTIFICATION ###
    imgpoints_l_undis = []
    imgpoints_r_undis = []

    for i, fname in enumerate(images_right):
        img_l = cv2.imread(images_left[i])
        img_r = cv2.imread(images_right[i])

        # convert to cv2
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # undistort
        img_l_undis = cv2.undistort(img_l, M1, d1, None, newCamMtx1)
        img_r_undis = cv2.undistort(img_r, M2, d2, None, newCamMtx2)

        # find the chess board corners in undistorted image
        ret_l_undis, corners_l_undis = cv2.findChessboardCorners(img_l_undis, (numEdgeX, numEdgeY), None)
        ret_r_undis, corners_r_undis = cv2.findChessboardCorners(img_r_undis, (numEdgeX, numEdgeY), None)

        if ret_l_undis is True:
            if ret_r_undis is True:
                rt = cv2.cornerSubPix(img_l_undis, corners_l_undis, (11, 11), (-1, -1), criteria)
                for j in range(0, len(rt)):
                    x = rt[j][0, :]
                    imgpoints_l_undis.append(x)

                rt = cv2.cornerSubPix(img_r_undis, corners_r_undis, (11, 11), (-1, -1), criteria)
                for j in range(0, len(rt)):
                    x = rt[j][0,:]
                    imgpoints_r_undis.append(x)

    # convert to np array
    imgpoints_l_undis = np.array(imgpoints_l_undis)
    imgpoints_r_undis = np.array(imgpoints_r_undis)

    # compute rectification uncalibrated
    ret, uncRectMtx1, uncRectMtx2 = cv2.stereoRectifyUncalibrated(imgpoints_l_undis, imgpoints_r_undis, F, img_shape)


    for i,fname in enumerate(images_right):
        ### REMAPPING ###
        # load images and convert to cv2 format
        img_l_name = images_left[i].split(os.sep)[-1]
        img_r_name = images_right[i].split(os.sep)[-1]

        img_l = cv2.imread(images_left[i])
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_l_undis = cv2.undistort(img_l, M1, d1, None, newCamMtx1)

        img_r = cv2.imread(images_right[i])
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        img_r_undis = cv2.undistort(img_r, M2, d2, None, newCamMtx2)

        # remap
        imglCalRect = cv2.remap(img_l, leftMapX, leftMapY, cv2.INTER_LINEAR)
        imgrCalRect = cv2.remap(img_r, rightMapX, rightMapY, cv2.INTER_LINEAR)
        numpyHorizontalCalibRect = np.hstack((imglCalRect, imgrCalRect))

        # warp for uncalibrated rectification
        imglUncalRect = cv2.warpPerspective(img_l_undis, uncRectMtx1, img_shape)
        imgrUncalRect = cv2.warpPerspective(img_r_undis, uncRectMtx2, img_shape)
        numpyHorizontalUncalibRect = np.hstack((imglUncalRect, imgrUncalRect))

        # # save
        # cv2.imwrite("imag/chessboard/left" + os.sep + img_l_name, img_l_undis)
        # cv2.imwrite("imag/chessboard/right" + os.sep + img_r_name, img_r_undis)
        # cv2.imwrite("imag/ImgCalRect" + os.sep + img_l_name, imglCalRect)
        # cv2.imwrite("imag/ImgCalRect" + os.sep + img_r_name, imgrCalRect)
        # cv2.imwrite("imag/ImgUncalRect" + os.sep + img_l_name, imglCalRect)
        # cv2.imwrite("imag/ImgUncalRect" + os.sep + img_r_name, imgrCalRect)

        print(img_l_name)
        print(img_r_name)

        ### SHOW RESULTS ###
        # calculate point arrays for epipolar lines
        lineThickness = 1
        lineColor = (255, 255, 0)
        numLines = 30
        interv = round(img_shape[0] / numLines)
        x1 = np.zeros((numLines, 1))
        y1 = np.zeros((numLines, 1))
        x2 = np.full((numLines, 1), (3*img_shape[1]))
        y2 = np.zeros((numLines, 1))
        for jj in range(0, numLines):
            y1[jj] = jj * interv
        y2 = y1

        for jj in range(0, numLines):
            cv2.line(numpyHorizontalCalibRect, (x1[jj], y1[jj]), (x2[jj], y2[jj]),
                     lineColor, lineThickness)
            cv2.line(numpyHorizontalUncalibRect, (x1[jj], y1[jj]), (x2[jj], y2[jj]),
                     lineColor, lineThickness)
            cv2.line(img_l,(x1[jj], y1[jj]), (x2[jj], y2[jj]),lineColor, lineThickness)
            cv2.line(img_r, (x1[jj], y1[jj]), (x2[jj], y2[jj]), lineColor, lineThickness)
            images = np.hstack((img_l, img_r))
        cv2.namedWindow("calibRect", cv2.WINDOW_NORMAL)
        cv2.namedWindow("uncalibRect", cv2.WINDOW_NORMAL)
        cv2.imshow("calibRect", numpyHorizontalCalibRect)
        cv2.imshow("uncalibRect", numpyHorizontalUncalibRect)
        cv2.imshow("left-right",images)
        cv2.waitKey()
        cv2.destroyAllWindows()

except (IOError, ValueError):
    print("An I/O error or a ValueError occurred")
except:
    print("An unexpected error occurred")
    raise