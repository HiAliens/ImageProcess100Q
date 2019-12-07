#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q12_Motion Filter.py
# time: 2019-11-27 12:08
# @Software: PyCharm
import numpy as np
import cv2
"""
[[1., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]]
"""
img = cv2.imread('imori.jpg')
cols, rows, c = img.shape
k_size = 3

# kernel
k = np.diag([1] * k_size).astype(np.float)
k /= k_size

# padding
pad = k_size // 2
out = np.zeros((2 * pad+cols, 2 * pad+rows, c), dtype=np.float)
out[pad:pad + cols, pad:pad + rows] = img.copy().astype(np.float)

for x in range(cols):
    for y in range(rows):
        for channel in range(c):
            out[pad+x, pad+y, channel] = np.sum(k * out[x:x+k_size, y:y+k_size, channel])

out = out[pad : pad + cols, pad : pad+rows].astype(np.uint8)

cv2.imwrite('Q12_out.jpg', out)
cv2.imshow('result', out)
cv2.waitKey(0)
cv2.destroyAllWindows()