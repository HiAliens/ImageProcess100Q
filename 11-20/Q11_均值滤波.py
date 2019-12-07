#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q11_均值滤波.py
# time: 2019-11-26 10:44
# @Software: PyCharm
import cv2
import numpy as np

img = cv2.imread('imori.jpg')
h, w, c = img.shape
k_size = 3
pad = k_size // 2
out = np.zeros((h+pad*2, w+pad*2, c), np.float)
out[pad:pad+w, pad:pad+h] = img.copy().astype(np.float)  # 周围补零完毕

for x in range(w):
    for y in range(h):
        for channel in range(c):
            out[y+pad, x+pad, channel] = np.mean(out[y:y+k_size, x:x+k_size, channel])

out = out[pad:pad+w, pad:pad+h].astype(np.uint8)

cv2.imwrite('Q11_out.jpg', out)
cv2.imshow('origin', img)
cv2.imshow('result', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
