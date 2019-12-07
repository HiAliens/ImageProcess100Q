#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q16_Prewitt滤波器.py
# time: 2019-12-01 8:50
# @Software: PyCharm
import cv2
import numpy as np

"""
卷积核参数：
v:  -1. -1. -1.
    0.  0.  0.
    1.  1.  1.

h:  -1. 0. 1.
    -1. 0. 1.
    -1. 0. 1.
"""


def bgr2gray(img):
    b = img[..., 0].copy()
    g = img[..., 1].copy()
    r = img[..., 2].copy()

    out = b * 0.0722 + g * 0.7125 + r * 0.2126
    out = out.astype(np.uint8)

    return out


def prewitt_filter(img, k_size=3):
    if img.shape == 3:
        w, h, c = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        w, h, c = img.shape

    pad = k_size // 2
    out = np.zeros((2 * pad + w, 2 * pad + h, c), dtype=np.float)
    out[pad: pad + w, pad: pad + h] = img.copy().astype(np.float)
    out_v = out.copy()
    out_h = out.copy()

    k_v = [[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]
    k_h = [[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]

    for x in range(w):
        for y in range(h):
            for channel in range(c):
                out_v[pad + y, pad + x, channel] = np.sum(k_v * (out[y: y + k_size, x: x + k_size, channel]))
                out_h[pad + y, pad + x, channel] = np.sum(k_h * (out[y: y + k_size, x: x + k_size, channel]))

    out_h = np.clip(out_h, 0, 255)
    out_v = np.clip(out_v, 0, 255)

    out_h = out_h[pad: pad + w, pad: pad + h].astype(np.uint8)
    out_v = out_v[pad: pad + w, pad: pad + h].astype(np.uint8)

    return out_h, out_v


if __name__ == '__main__':
    img = cv2.imread('imori.jpg')
    img_gray = bgr2gray(img)
    out_h, out_v = prewitt_filter(img_gray)

    cv2.imwrite('Q16_h_out.jpg', out_h)
    cv2.imwrite('Q16_v_out.jpg', out_v)
    cv2.imshow('out_h', out_h)
    cv2.imshow('out_v', out_v)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


