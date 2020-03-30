#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/28 17:27
# @Author  : huangkai
# @Site    : 
# @File    : main.py
# @Software: PyCharm

import numpy as np
import os
import cv2 as cv
import glob

B = np.load('B.npz')
# print("\nM1 =")
# print(B['M1'])
# print("\nd1 =")
# print(B['d1'])
# print("\nM2 =")
# print(B['M2'])
# print("\nd1 =")
# print(B['d1'])
# print("\nR =")
# print(B['R'])
# print("\nT =")
# print(B['T'])
print("\nnewCamMtx1 =")
print(B['newCamMtx1'])
print("\nnewCamMtx2 =")
print(B['newCamMtx2'])

#
# B = np.load('B_r.npz')
# print("\nmtx =")
# print(B['mtx'])
# print("\ndist =")
# print(B['dist'])
# print("\nrvecs =")
# print(B['rvecs'])
# print("\ntvecs =")
# print(B['tvecs'])