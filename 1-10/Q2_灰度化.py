#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: 灰度化.py
# time: 2019-11-17 19:53
# @Software: PyCharm

import cv2
import numpy as np

img1 = cv2.imread('imori.jpg')

img = cv2.imread('imori.jpg').astype(np.float)  # 读取数据时就转换程浮点数类型

b = img[:, :, 0].copy()  # python 浅层复制，只复制父对象，不复制子对象
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# img[:] = [b, g, r]
out = 0.2126 * r + 0.7152 * g + 0.0722 * b
out = out.astype(np.uint8)  # 计算过后恢复成整型

cv2.imwrite('Q2_out.jpg', out)
cv2.imshow('gray', out)
cv2.imshow('origin', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
