#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q19_log滤波器.py
# time: 2019-12-04 9:31
# @Software: PyCharm
import cv2
import numpy as np
"""
类似于高斯滤波器的卷积核表示方法，但是具体公式不同,

"""


def bgr2gray(img):
    b = img[..., 0].copy()
    g = img[..., 1].copy()
    r = img[..., 2].copy()

    out = 0.0722 * b + 0.7125 * g + 0.2126 * r
    out = out.astype(np.uint8)

    return out


def log_filter(img, k_size=3, sigma=3):
    h, w = img.shape
    pad = k_size // 2
    out = np.zeros((2 * pad + w, 2 * pad + h), dtype=np.float)
    out[pad: pad + w, pad: pad + h] = img.copy().astype(np.float)
    tem = out.copy()

    k = np.zeros((k_size, k_size), dtype=np.float)
    for x in range(-pad, -pad + k_size):
        for y in range(-pad, -pad + k_size):
            k[pad + x, pad + y] = (x ** 2 + y ** 2 - sigma ** 2) * np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    k /= (2 * np.pi * (sigma ** 6))
    k /= k.sum()

    for x in range(w):
        for y in range(h):
            out[pad + x, pad + y] = np.sum(k * out[x: x + k_size, y: y+k_size])
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


if __name__ == '__main__':
    img = cv2.imread('imori.jpg')
    gray = bgr2gray(img)
    result = log_filter(gray)

    cv2.imwrite('Q19_out.jpg', result)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()