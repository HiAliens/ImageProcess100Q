#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q13_MAX-MIN滤波器.py
# time: 2019-11-28 15:00
# @Software: PyCharm
import cv2
import numpy as np


def bgr2gray(img):
    b = img[..., 0].copy()
    g = img[..., 1].copy()
    r = img[..., 2].copy()

    # Gray scale
    out = 0.0722 * b + 0.7125 * g + 0.2126 * r
    out = out.astype(np.uint8)

    return out


def max_min_filter(img, k_size=3):
    w, h = img.shape
    pad = k_size // 2
    out = np.zeros((w + 2 * pad, h + 2 * pad), dtype=np.float)
    out[pad:pad+w, pad:pad+h] = img.copy().astype(np.float)
    for x in range(w):
        for y in range(h):
                out[pad+y, pad+x] = np.max(out[y: y + k_size, x: x + k_size]) \
                                             - np.min(out[y: y + k_size, x: x + k_size])

    out = out[pad:pad+w, pad:pad+h].astype(np.uint8)
    return out


if __name__ == '__main__':
    img = cv2.imread('imori.jpg')
    img_gray = bgr2gray(img)
    result = max_min_filter(img_gray)
    cv2.imwrite('Q13_out.jpg', result)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()