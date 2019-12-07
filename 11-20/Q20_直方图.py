#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q20_直方图.py
# time: 2019-12-06 13:56
# @Software: PyCharm
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("imori.jpg").astype(np.float)

plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("Q20_out.png")
plt.show()
