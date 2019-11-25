#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: 大津二值化算法.py
# time: 2019-11-19 10:10
# @Software: PyCharm

import cv2
import numpy as np

img = cv2.imread('imori.jpg').astype(np.float)
img_gray = 0.0722 * img[..., 0] + 0.7125 * img[..., 1] + 0.2126 * img[..., 2]
out = img_gray.astype(np.uint8)

max_sigma = 0  # 最大类间方差
max_t = 0  # threshold

h, w, c = img.shape
for _t in range(0, 255):
    v0 = out[np.where(out < _t)]
    m0 = np.mean(v0)if len(v0) > 0 else 0
    r0 = len(v0) / (h * w)
    v1 = out[np.where(out >= _t)]
    m1 = np.mean(v1)if len(v1) > 0 else 0
    r1 = len(v1) / (h * w)
    sigma = r0 * r1 * ((m1 - m0) ** 2)
    if sigma > max_sigma:
        max_sigma = sigma
        max_t = _t

print('threshold >> ', max_t)
out[out < max_t] = 0
out[out >= max_t] = 255

cv2.imwrite('big_jin_binary.jpg', out)
cv2.imshow('out', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
