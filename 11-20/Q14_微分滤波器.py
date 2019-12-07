#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q14_微分滤波器.py
# time: 2019-11-29 13:54
# @Software: PyCharm
"""
使用两个矩阵作为卷积核，分别在垂直和水平方向上进行运算
水平：
 0 0 0
-1 1 0
 0 0 0
垂直：
0 -1 0
0  1 0
0  0 0
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def bgr2gray(img):
    b = img[..., 0].copy()
    g = img[..., 1].copy()
    r = img[..., 2].copy()

    out = b * 0.722 + g * 0.7125 + r * 0.2126
    out.astype(np.uint8)
    return out


def differential_fiter(img, k_size = 3):
    w, h = img.shape
    pad = k_size // 2
    out = np.zeros((2 * pad + w, 2 * pad + h), dtype=np.float)
    out[pad: pad + w, pad:  pad + h] = img.copy().astype(np.float)
    k_v = [[0., -1., 0.], [0., 1., 0.], [0., 0., 0.]]
    k_h = [[0., 0., 0.], [-1., 1., 0.], [0., 0., 0.]]
    out_v = out.copy()
    out_h = out.copy()
    for x in range(w):
        for y in range(h):
            out_v[pad + x, pad + y] = np.sum(k_v * (out[x: x + k_size, y: y + k_size]))
            out_h[pad + x, pad + y] = np.sum(k_h * (out[x: x + k_size, y: y + k_size]))
    out_h = np.clip(out_h, 0, 255)
    out_v = np.clip(out_v, 0, 255)

    return out_v, out_h


if __name__ == '__main__':
    img = cv2.imread('imori.jpg')
    img_gray = bgr2gray(img)
    out_v, out_h = differential_fiter(img_gray)

    cv2.imwrite('Q14_v_out.jpg', out_v)
    cv2.imwrite('Q14_h_out.jpg', out_h)

    titles = ['out_v', 'out_h']
    images = [out_v, out_h]

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
        plt.imshow(images[i], 'gray')
plt.show()