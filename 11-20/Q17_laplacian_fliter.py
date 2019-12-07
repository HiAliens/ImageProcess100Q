#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q17_laplacian_fliter.py
# time: 2019-12-02 9:46
# @Software: PyCharm

"""
 0  1  0
 1 -4 1
 0  1 0
"""
import cv2
import numpy as np


def bgr2gray(img):
    b = img[..., 0].copy()
    g = img[..., 1].copy()
    r = img[..., 2].copy()

    out = 0.0722 * b + 0.7152 * g + 0.2126 * r
    out = out.astype(np.uint8)

    return out


def laplacian(gray_img, k_size=3):
    h, w = gray_img.shape
    pad = k_size // 2
    out = np.zeros((2 * pad + h, 2 * pad + w), dtype=np.float)
    out[pad: pad + h, pad: pad + w] = gray_img.copy().astype(np.float)
    K = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]

    tem = out.copy()  # 必须转换

    for y in range(h):
        for x in range(w):
            out[pad + y, pad + x] = np.sum(K * (tem[y: y + k_size, x: x + k_size]))
    # print(out)
    out = np.clip(out, 0, 255)
    out = out[pad: pad + h, pad: pad + w].astype(np.uint8)

    return out


img = cv2.imread('imori.jpg').astype(np.float)
gray_img = bgr2gray(img)
result = laplacian(gray_img)


cv2.imwrite('Q17_out.jpg', result)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()