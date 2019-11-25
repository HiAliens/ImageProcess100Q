#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q7_平均池化.py
# time: 2019-11-22 10:25
# @Software: PyCharm
import cv2
import numpy as np

img = cv2.imread('imori.jpg').astype(np.float)
kernel_size = 8
result = img.copy()

_sum = 0
for i in range(3):
    tem = img[..., i].copy()
    for j in range(16):
        for k in range(16):
            for row in range(kernel_size*j, kernel_size*(j+1)):
                for col in range(kernel_size*k, kernel_size*(k+1)):
                    _sum += tem[row][col]
            result[..., i][j][k] = (_sum / 64).astype(np.int)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()