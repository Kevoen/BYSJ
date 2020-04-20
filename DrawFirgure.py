#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 17:14
# @Author  : huangkai
# @Site    : 
# @File    : DrawFirgure.py
# @Software: PyCharm
import  cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

def drawScatters(point,color):
    fig = plt.figure()
    axes3d = Axes3D(fig)
    axes3d.scatter(point[:,0], point[:, 1], point[:, 2], c=color,s=8)
    plt.show()

if __name__ == '__main__':
    output = np.load('output_points.npz')
    output_c = np.load('output_colors.npz')
    points = output['points']
    colors = output_c['colors']
    max = points[:,0].size
    rands = np.random.randint(0,max,size=4000)
    point = points[rands,:]
    color = colors[rands,:]*0.001
    print(max)
    drawScatters(points,colors*0.001)