#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Author : zxiaosi
# @Time : 2022/7/27 11:12
# @desc : 三种梯度图处理方法
# 参考：https://blog.csdn.net/Ms_X_Zi/article/details/96424896
import cv2
import numpy as np


def grad_map_sobel(img_flow: np.ndarray) -> np.ndarray:
    """
    图片梯度处理(边缘检测) -- Sobel滤波器

    :param img_flow: 图片流
    :return: 图片流
    """

    """
    cv2.Sobel 滤波器
    :param src: 图片流
    :param ddepth: 输出图片深度, -1:输出原图片深度
    :param dx: 横向滤波器类型
    :param dy: 纵向滤波器类型
    :param ksize: 滤波器大小, 3, 5, 7
    :param scale: 放缩比例
    :param delta: 偏差
    :param borderType: 边界处理方式
    :return: 图片流
    """
    grad_x = cv2.Sobel(img_flow, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(img_flow, cv2.CV_64F, 0, 1)

    """
    cv2.convertScaleAbs 放缩
    :param src: 图片流
    :param alpha: 放缩比例
    :param beta: 偏差
    :return: 图片流
    """
    grad_abs_x = cv2.convertScaleAbs(grad_x)
    grad_abs_y = cv2.convertScaleAbs(grad_y)

    """
    cv2.addWeighted 混合
    :param src1: 图片流1
    :param alpha: 放缩比例1
    :param src2: 图片流2
    :param beta: 放缩比例2
    :param gamma: 偏差
    :return: 图片流
    """
    grad_xy = cv2.addWeighted(grad_abs_x, 0.5, grad_abs_y, 0.5, 0)

    return grad_xy


def grad_map_scharr(img_flow: np.ndarray) -> np.ndarray:
    """
    图片梯度处理(边缘检测) -- Scharr滤波器

    :param img_flow: 图片流
    :return: 图片流
    """

    """
    cv2.Scharr 滤波器
    :param src: 图片流
    :param ddepth: 输出图片深度, -1:输出原图片深度
    :param dx: 横向滤波器类型
    :param dy: 纵向滤波器类型
    :param scale: 放缩比例
    :param delta: 偏差
    :param borderType: 边界处理方式
    :return: 图片流
    """
    grad_x = cv2.Scharr(img_flow, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(img_flow, cv2.CV_64F, 0, 1)

    """
    cv2.convertScaleAbs 放缩
    :param src: 图片流
    :param alpha: 放缩比例
    :param beta: 偏差
    :return: 图片流
    """
    grad_abs_x = cv2.convertScaleAbs(grad_x)
    grad_abs_y = cv2.convertScaleAbs(grad_y)

    """
    cv2.addWeighted 混合
    :param src1: 图片流1
    :param alpha: 放缩比例1
    :param src2: 图片流2
    :param beta: 放缩比例2
    :param gamma: 偏差
    :return: 图片流
    """
    grad_xy = cv2.addWeighted(grad_abs_x, 0.5, grad_abs_y, 0.5, 0)

    return grad_xy


def grad_map_laplacian(img_flow: np.ndarray) -> np.ndarray:
    """
    图片梯度处理(边缘检测) -- Laplacian滤波器

    :param img_flow: 图片流
    :return: 图片流
    """

    """
    cv2.Laplacian 滤波器
    :param src: 图片流
    :param ddepth: 输出图片深度, -1:输出原图片深度
    :param ksize: 滤波器大小, 3, 5, 7
    :param scale: 放缩比例
    :param delta: 偏差
    :param borderType: 边界处理方式
    :return: 图片流
    """
    lap = cv2.Laplacian(img_flow, cv2.CV_32F)

    """
    cv2.convertScaleAbs 放缩
    :param src: 图片流
    :param alpha: 放缩比例
    :param beta: 偏差
    :return: 图片流
    """
    grad_xy = cv2.convertScaleAbs(lap)

    return grad_xy
