#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q9_高斯滤波.py
# time: 2019-11-24 9:57
# @Software: PyCharm
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('imori_noise.jpg')
H, W, C = img.shape

k_size = 3
sigm = 1.3
pad = k_size // 2
out = np.zeros((H + 2 * pad, W + 2 * pad, C), np.float)
out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)  # 这里只制定了out的x,y

k = np.zeros((k_size, k_size), np.float)

# 卷积核计算，以中心点为（0，0），其余点上下左右，y、x加减1，以pad为最远计算距离
for x in range(-pad, -pad + k_size):
    for y in range(-pad, -pad + k_size):
        k[pad+y, pad+x] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigm ** 2)))
    k /= (sigm * np.square(2 * np.pi))
    k /= k.sum()

tem = out.copy()
# 卷积计算
for x in range(H):
    for y in range(W):
        for c in range(C):
            out[y+pad, x+pad, c] = np.sum(k * tem[y:y+k_size, x:x+k_size, c])

# 恢复图像原始大小
out = out[pad:pad+H, pad:pad+W].astype(np.uint8)
cv2_Gs = cv2.GaussianBlur(img, (3, 3), 1.3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

titles = ['origin', 'cv2_Gs', 'my_Gs']
result = [img, cv2_Gs, out]

cv2.imwrite('Q9_out.jpg', out)
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(titles[i])
    plt.imshow(result[i])
    plt.xticks([]), plt.yticks([])
plt.show()
