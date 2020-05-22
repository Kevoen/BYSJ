#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/9 18:47
# @Author  : huangkai
# @Site    : 
# @File    : capturecamera.py
# @Software: PyCharm

import numpy as np
import cv2

cap = cv2.VideoCapture(2)

while(True):

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()