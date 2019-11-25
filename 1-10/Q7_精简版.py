#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q7_精简版.py
# time: 2019-11-22 11:20
# @Software: PyCharm
import cv2
import numpy as np
'''
没有改变图像的大小
'''
# Read image
img = cv2.imread("imori.jpg")

# Average Pooling
out = img.copy()

H, W, C = img.shape
G = 8
Nh = int(H / G)
Nw = int(W / G)

for y in range(Nh):
    for x in range(Nw):
        for c in range(C):
            out[G*y:G*(y+1), G*x:G*(x+1), c] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1), c]).astype(np.int)

# Save result
cv2.imwrite("Q7_out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()