#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q18_Emboss滤波器.py
# time: 2019-12-03 14:14
# @Software: PyCharm
import cv2
import numpy as np
"""
卷积核
[[-2., -1., 0.],
[-1., 1., 1.], 
[0., 1., 2.]]
"""


def brg2gray(img):
    b = img[..., 0].copy()
    g = img[..., 1].copy()
    r = img[..., 2].copy()

    out = b * 0.0722 + g * 0.7125 + r * 0.2126
    out = out.astype(np.uint8)

    return out


def Emboss_fliter(gray_img, k_size=3):
    h, w = gray_img.shape
    pad = k_size // 2
    out = np.zeros((2 * pad + h, 2 * pad + w), dtype=np.float)
    out[pad: pad + w, pad: pad + h] = gray_img.copy().astype(np.float)
    k = [[-2., -1., 0.], [-1., 1., 1.], [0., 1., 2.]]

    tem = out.copy()

    for x in range(w):
        for y in range(h):
            out[pad + x, pad + y] = np.sum(k * (tem[x: x + k_size, y: y + k_size]))

    out = np.clip(out, 0, 255)
    out = out[pad: pad + w, pad: pad + h].astype(np.uint8)

    return out


if __name__ == '__main__':
    img = cv2.imread('imori.jpg')
    gray = brg2gray(img)
    result = Emboss_fliter(gray)

    cv2.imwrite('Q18_out.jpg', result)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

