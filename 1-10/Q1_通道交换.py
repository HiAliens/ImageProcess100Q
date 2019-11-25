#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: 通道交换.py
# time: 2019-11-16 18:48
# @Software: PyCharm

import cv2

img = cv2.imread('imori.jpg')
B = img[:, :, 0].copy()  # 必须使用copy（），不然结果不对
G = img[:, :, 1].copy()
R = img[:, :, 2].copy()
# img[:] = [R, G, B]

img[:, :, 0] = R
img[:, :, 1] = G
img[:, :, 2] = B

cv2.imwrite('Q1_OUT.jpg', img)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()