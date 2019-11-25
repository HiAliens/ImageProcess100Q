#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
# file: Q5_HSV变换.py
# time: 2019-11-20 10:26
# @Software: PyCharm

import cv2
import numpy as np

"""
在做图像处理的时候，脑子里应有这样一个概念：我们需要处理图像中每一个像素点的数值，
对于彩色图像，要想象出来有3层网格.
经常出错的点：
1.有条件运算的时候，总是忘记通过索引去寻找符合条件的像素值，错误地直接使用H、S、V去运算，结果形状不匹配
2.计算时候错误地使用广播机制，造成数组形状不匹配，报错。
"""
img = cv2.imread('imori.jpg').astype(np.float) / 255.
# RGB > HSV
out = np.zeros_like(img)

max_v = np.max(img, axis=2).copy()  # 二维数组,每个像素点（bgr三通道中）的最大值
min_v = np.min(img, axis=2).copy()
min_arg = np.argmin(img, axis=2)  # 最小值的坐标
print('the shape of max_v is ', max_v.shape)

B = img[..., 0].copy()
G = img[..., 1].copy()
R = img[..., 2].copy()

H = np.zeros_like(max_v)
H[np.where(min_v == max_v)] = 0

idx = np.where(min_arg == 0)  # 通过坐标去索引值，公式中是去寻找最小值，值于坐标相对应，找到了坐标，也就找到了值
H[idx] = 60 * (G[idx] - R[idx]) / (max_v[idx] - min_v[idx]) + 60

idx = np.where(min_arg == 2)
H[idx] = 60 * (B[idx] - G[idx]) / (max_v[idx] - min_v[idx]) + 180

idx = np.where(min_arg == 1)
H[idx] = 60 * (R[idx] - B[idx]) / (max_v[idx] - min_v[idx]) + 300

V = max_v.copy()
S = max_v.copy() - min_v.copy()

H = (180 + H) % 360  # 别忘记取余，最大为360

C = S
H_ = H / 60
X = C * (1 - np.abs(H_ % 2 - 1))
Z = np.zeros_like(H)
print(H_.shape)
'''
这里使用了列表将需要计算的像素矩阵囊括了进去，目的是在for循环中，通过变量i去索引，复用代码。
'''
add_space = [[C, X, Z], [X, C, Z], [Z, C, X], [Z, X, C], [X, Z, C], [C, Z, X]]
diff = V - C
for i in range(6):
    idx = np.where((i <= H_) & (H_ < (i + 1)))  # 这里要使用位运算符，不能使用逻辑运算符 and
    out[..., 2][idx] = diff[idx] + add_space[i][0][idx]
    out[..., 1][idx] = diff[idx] + add_space[i][1][idx]
    out[..., 0][idx] = diff[idx] + add_space[i][2][idx]

out[np.where(max_v == min_v)] = 0

cv2.imwrite('Q5_output.jpg', out)
cv2.imshow('result', out)
cv2.waitKey(0)
cv2.destroyAllWindows()