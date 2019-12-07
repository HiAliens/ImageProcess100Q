#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q15_sobel滤波器.py
# time: 2019-11-30 9:51
# @Software: PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt


def bgr2gray(img):
    b = img[..., 0].copy()
    g = img[..., 1].copy()
    r = img[..., 2].copy()

    out = b * 0.0772 + g * 0.7125 + r * 0.2126
    out = out.astype(np.uint8)

    return out


def sobel_fliter(img, k_size=3):
    if img.shape == 3:
        w, h, c = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        w, h, c = img.shape

    pad = k_size // 2
    out = np.zeros((2 * pad + w, 2 * pad + h, c), dtype=np.float)
    out[pad: pad + w, pad: pad + h] = img.copy().astype(np.float)
    k_v = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
    k_h = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]

    out_v = out.copy()
    out_h = out.copy()

    for x in range(w):
        for y in range(h):
            for channel in range(c):
                out_v[pad + x, pad + y, channel] = np.sum(k_v * (out[x: x + k_size, y: y + k_size, channel]))
                out_h[pad + x, pad + y, channel] = np.sum(k_h * (out[x: x + k_size, y: y + k_size, channel]))

    out_v = np.clip(out_v, 0, 255)
    out_h = np.clip(out_h, 0, 255)

    out_h = out_h[pad: pad + w, pad: pad + h].astype(np.uint8)
    out_v = out_v[pad: pad + w, pad: pad + h].astype(np.uint8)

    return out_v, out_h


if __name__ == '__main__':
    img = cv2.imread('imori.jpg').astype(np.float)
    img_gray = bgr2gray(img)
    out_v, out_h = sobel_fliter(img_gray)

    cv2.imshow('out_h', out_h)
    cv2.imshow('out_v', out_v)

    cv2.waitKey(0)
    cv2.destroyAllWindows()