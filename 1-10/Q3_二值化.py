#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: 二值化.py
# time: 2019-11-18 11:18
# @Software: PyCharm

import cv2
import numpy as np

"""
首先利用上节课内容将图像灰度化，之后再对图像进行二值化
"""
img = cv2.imread('imori.jpg').copy().astype(np.float)
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]
out = r * 0.2126 + g * 0.7125 + b * 0.0722
out = out.astype(np.uint8)

# print(out>127)
out[out < 127] = 0  # 只对真值进行赋值
out[out >= 127] = 255

cv2.imwrite('Q3_out.jpg', out)
cv2.imshow('binary', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
