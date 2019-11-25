#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q6_减色处理.py
# time: 2019-11-21 10:00
# @Software: PyCharm
import cv2
import numpy as np

img = cv2.imread('imori.jpg').copy()
# B = img[..., 0].copy()
# G = img[..., 1].copy()
# R = img[..., 2].copy()

out = np.zeros_like(img)  # 存储处理后的图像像素

idx = np.where((0 <= img) & (img <= 64))
out[idx] = 32

idx = np.where((64 <= img) & (img <= 128))
out[idx] = 96

idx = np.where((128 <= img) & (img <= 192))
out[idx] = 160

idx = np.where((192 <= img) & (img <= 256))
out[idx] = 224

'''
代码简化：
out = img // 64 * 64 + 32  # 这是先观察，再思考，再去做的例子
'''

cv2.imwrite('Q6_output.jpg', out)
cv2.imshow('out', out)
cv2.waitKey(0)
cv2.destroyAllWindows()