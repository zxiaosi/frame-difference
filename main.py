#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Author : zxiaosi
# @Time : 2022/7/26 14:03
# @desc : 主函数
import os

import cv2
import numpy as np

from loguru import logger

from config import setting
from gradient_map import grad_map_sobel, grad_map_scharr, grad_map_laplacian
from utils import show_picture, read_picture, save_picture, get_picture_attribute, scale_picture, binarization, \
    write_text_on_picture


def gradient_map(img_flow: np.ndarray, choice: int = 1):
    match choice:
        case 1:
            grad_xy = grad_map_sobel(img_flow)
        case 2:
            grad_xy = grad_map_scharr(img_flow)
        case 3:
            grad_xy = grad_map_laplacian(img_flow)
        case _:
            logger.error("gradient_map() takes argument error!!!")
            return None

    return grad_xy


def handle_image(img_path: str) -> np.ndarray:
    """
    处理图片

    :param img_path: 图片路径
    :return: 图片流
    """

    gray_image = read_picture(path=img_path, flag=0)  # 读取图片(灰度图)

    get_picture_attribute(img_flow=gray_image)  # 得到图片的属性

    scale_image = scale_picture(gray_img=gray_image, scale=setting.ZOOM_RATIO)  # 将图片缩小2倍

    # 图片梯度处理, choice: 1-sobel, 2-scharr, 3-laplacian
    grad_image = gradient_map(img_flow=scale_image, choice=setting.GRAD_MAP_CHOICE)

    bin_image = binarization(image_flow=grad_image)  # 二值化

    return bin_image


def abs_diff_image(img_path1: str, img_path2: str) -> np.ndarray:
    """
    求两张图片的绝对差

    :param img_path1: 图片1路径
    :param img_path2: 图片2路径
    :return: 图片流
    """

    assert os.path.exists(img_path1) and os.path.exists(img_path1), "image_path1 和 image_path2 不存在"

    image_flow1 = handle_image(img_path1)
    image_flow2 = handle_image(img_path2)

    assert image_flow1.shape == image_flow2.shape, "两张图片大小不一致, 请调整图片大小"

    """
    cv2.absdiff
    :param src1: 图片流1
    :param src2: 图片流2
    :return: 图片流
    """
    abs_img = cv2.absdiff(image_flow1, image_flow2)

    return abs_img


def frame_difference(img_flow: np.ndarray, diff_thresh: float = 0.5) -> np.ndarray:
    """
    对绝对差图片做处理

    :param img_flow: 帧差图片
    :param diff_thresh: 帧差阈值
    """

    """
    cv2.getStructuringElement
    :param shape: 形状, cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS
    :param ksize: 形状大小, eg: (5, 5)
    :return: 形状
    """
    # 图像形态学处理 -- https://blog.csdn.net/duwangthefirst/article/details/80001590
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    """
    cv2.morphologyEx
    :param src: 图片流
    :param op: 操作, cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_GRADIENT, cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT
    :param kernel: 形状
    :param anchor: 锚点
    :param iterations: 迭代次数
    :param borderType: 边界类型, cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP
    :return: 图片流
    """
    # 对图像进行一系列的膨胀腐蚀组合 -- https://www.jianshu.com/p/ee72f5215e07
    diff_img = cv2.morphologyEx(img_flow, cv2.MORPH_OPEN, kernel)  # 形态学操作

    """
    cv2.meanStdDev
    :param src: 图片流
    :param mean: 平均值
    :param stddev: 标准差
    :param mask: 掩码
    """
    mean, stddev = cv2.meanStdDev(src=diff_img)

    logger.success(f'差异度: {mean}, 两张图片是否存在差异: {"Yes" if mean[0][0] > diff_thresh else "No"}')

    return diff_img


if __name__ == '__main__':
    abs_image = abs_diff_image(setting.FIRST_PIC_INPUT, setting.SECOND_PIC_INPUT)  # 求两张图片的绝对差

    diff_image = frame_difference(abs_image, setting.DIFF_THRESH)  # 对绝对差图片做处理

    save_picture(diff_image, setting.OUTPUT)  # 保存图片 (./images/result.png)

    # write_image = write_text_on_picture(diff_image, "Hello World!", (100, 100))  # 写入图片
    #
    # show_picture(write_image, "Frame Difference", 5000)  # 显示图片 (任意键关闭 或 5s后关闭)
