#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/3 14:33
# @Author  : huangkai
# @Site    : 
# @File    : SGBM.py
# @Software: PyCharm
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# imgL = cv2.imread('imag/teddy/im2.ppm')
# imgR = cv2.imread('imag/teddy/im6.ppm')
#
# grayLeft = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
# grayRight = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
#
# stereo = cv2.StereoBM_create(numDisparities=64,blockSize=11)
# # stereo = cv2.StereoSGBM_create(numDisparities=64,blockSize=10)
# # disparity = stereo.compute(imgL,imgR)
# disparity = stereo.compute(grayLeft,grayRight)
# plt.imshow(disparity,'gray')
# plt.show()


import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

# Set disparity parameters
# Note: disparity range is tuned according to specific parameters obtained through trial and error.
win_size = 5
min_disp = -1
max_disp = 63  # min_disp * 9
num_disp = max_disp - min_disp  # Needs to be divisible by 16

# Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')



# Create Block matching object.
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=11,
                               uniquenessRatio=10,
                               speckleWindowSize=10,
                               speckleRange=10,
                               disp12MaxDiff=1,
                               P1=8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                               P2=32 * 3 * win_size ** 2)  # 32*3*win_size**2)

# Compute disparity map
print("\nComputing the disparity  map...")

tic1=time.time()
imgL = cv2.imread('imag/teddy/im2.ppm')
imgR = cv2.imread('imag/teddy/im6.ppm')
# imgL = cv2.imread('imag/mychessborad/bor01.jpg') # queryImage
# imgR = cv2.imread('imag/mychessborad/bor02.jpg') # trainImage

disparity = stereo.compute(imgL, imgR)
# Show disparity map before generating 3D cloud to verify that point cloud will be usable.
size1, size2 = disparity.shape

B = np.load("B.npz")
M = B['M1']
f = M[0,0]
print(M,f,disparity)
depth=np.ones_like(disparity,dtype=np.uint8)
for i in range(size1):
    for j in range(size2):
        if abs(disparity[i][j])<5: ##噪音
            depth[i][j]=0
        else:
            depth[i][j]=f*40/disparity[i][j]

print(depth)
plt.title('disparity and depth')
plt.subplot(221), plt.imshow(imgL)
plt.subplot(222), plt.imshow(imgR)
plt.subplot(223), plt.imshow(disparity)
plt.subplot(224), plt.imshow(depth)
print('用时:',time.time()-tic1)
plt.show()

# # Generate  point cloud.
# print ("\nGenerating the 3D map...")
#
# # Get new downsampled width and height
# h,w = imgL.shape[:2]
#
# # Load focal length.
# focal_length = f/100
#
# # Perspective transformation matrix
# # This transformation matrix is from the openCV documentation, didn't seem to work for me.
# Q = np.float32([[1,0,0,-w/2.0],
#     [0,-1,0,h/2.0],
#     [0,0,0,-focal_length],
#     [0,0,1,0]])
# # This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision.
# # Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
# Q2 = np.float32([[1,0,0,0],
#     [0,-1,0,0],
#     [0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally.
#     [0,0,0,1]])
# # Reproject points into 3D
# points_3D = cv2.reprojectImageTo3D(disparity, Q2)
#
# # Get color points
# colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
#
# # Get rid of points with value 0 (i.e no depth)
# mask_map = disparity > disparity.min()
#
# # Mask colors and points.
# output_points = points_3D[mask_map]
# output_colors = colors[mask_map]
# np.savez('output_points.npz', points = output_points)
# np.savez('output_colors.npz', colors = output_colors)
# # Define name for output file
# output_file = 'reconstructed.ply'
#
# # Generate point cloud
# print ("\n Creating the output file... \n")
# create_output(output_points, output_colors, output_file)

