#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q10_中值滤波.py
# time: 2019-11-25 14:02
# @Software: PyCharm
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imori_noise.jpg')
h, w, c = img.shape
k_size = 3  # 卷积核大小
pad = k_size // 2  # 补零的行数
out = np.zeros((h + 2 * pad, w + 2 * pad, c), np.float)  # 补零后的图像
out[pad: pad + h, pad: pad + w] = img.copy().astype(np.float)

for y in range(h):
    for x in range(w):
        for chanel in range(c):
            out[pad+y, pad+x, chanel] = np.median(out[y:y+k_size, x:x+k_size, chanel])

out = out[pad: pad + h, pad: pad + w].astype(np.uint8)

out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

titles = ['origin', 'result']
images = [img, out_rgb]
cv2.imwrite('Q10_out.jpg', out)
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(titles[i])
    plt.imshow(images[i])
    plt.xticks([]), plt.yticks([])
plt.show()
