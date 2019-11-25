#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q8_最大池化.py
# time: 2019-11-23 13:36
# @Software: PyCharm
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('imori.jpg')
out = img.copy()

kenel_size = 8
H, W, C = img.shape
h = H // kenel_size
w = W // kenel_size

for i in range(h):
    for j in range(w):
        for c in range(C):
            out[kenel_size*i:kenel_size*(i+1), kenel_size*j:kenel_size*(j+1), c] \
                = np.max(out[kenel_size*i:kenel_size*(i+1), kenel_size*j:kenel_size*(j+1), c])

cv2.imwrite('Q8_out.jpg', out)
result = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
titles = ['origin', 'max_pooling']
images = [img, result]
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()