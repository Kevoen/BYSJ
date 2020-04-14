#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/26 13:58
# @Author  : huangkai
# @Site    : 
# @File    : Calibration.py
# @Software: PyCharm
import numpy as np
import cv2 as cv
import glob
import os
import PIL.ExifTags
import PIL.Image

def calib_n(inter_corner_shape, img_dir, img_type):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    w, h = inter_corner_shape

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((h * w, 3), np.float32)
    objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # images = glob.glob('imag/chessboard/*.jpg')
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    gray = None
    for fname in images:
        img = cv.imread(fname)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (w, h), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            print(fname,"successful!!")
            # # Draw and display the corners
            # cv.drawChessboardCorners(img, (w, h), corners2, ret)
            # cv.imshow('img', img)
            # cv.waitKey()
        else:
            print(fname)
    cv.destroyAllWindows()

    # calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # Save parameters into numpy file
    np.save("./camera_params/ret", ret)
    np.save("./camera_params/K", mtx)
    np.save("./camera_params/dist", dist)
    np.save("./camera_params/rvecs", rvecs)
    np.save("./camera_params/tvecs", tvecs)


    print(("ret:"), ret)
    print(("internal matrix:\n"), mtx)
    # in the form of (k_1,k_2,p_1,p_2,k_3)
    print(("distortion cofficients:\n"), dist)
    # R,T
    print(("rotation vectors:\n"), rvecs)
    print(("translation vectors:\n"), tvecs)

    # calculate the error of reproject
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )

    return mtx, dist


def dedistortion(img_dir, img_type, save_dir, mtx, dist):
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img_name = fname.split(os.sep)[-1]
        img = cv.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv.imwrite(save_dir + os.sep + img_name, dst)
    print('Dedistorted images have been saved to: %s successfully.' % save_dir)


if __name__ == '__main__':
    inter_corner_shape = (7, 5)
    img_dir = "imag/chessboard"
    # img_dir = "3DReconstruction/Calibration/calibration_images"
    img_type = "jpg"
    mtx, dist= calib_n(inter_corner_shape, img_dir, img_type)
    # save_dir = "./imag/save_dedistortion"
    # if (not os.path.exists(save_dir)):
    #     os.makedirs(save_dir)
    # dedistortion(img_dir, img_type, save_dir, mtx, dist)
    # calib_n(inter_corner_shape, save_dir, img_type)