#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Author : zxiaosi
# @Time : 2022/7/26 14:07
# @desc : 工具包
import os

import cv2
import numpy as np

from loguru import logger

from config import setting


def get_picture_attribute(img_flow: np.ndarray):
    """
    得到图片的属性

    :param img_flow: 图片流
    """
    assert img_flow is not None, "图片流不能为空"

    height, width, *rest = img_flow.shape  # 图片的行、列、通道数 (灰度图没有通道)
    logger.info(f"图片的行、列、通道数: {height, width, *rest}, 图片的类型: {img_flow.dtype}, 图片的大小: {img_flow.size}kb")


def read_picture(path: str, flag: int = 1) -> np.ndarray:
    """
    将图片转为图片流

    :param path: 图片路径
    :param flag: 1:彩色图片，0:灰度图片（默认彩色图）
    :return: 图片流
    """
    assert os.path.exists(path), "图片必须存在"

    """ 
    cv2.imread 读取图片
    :param filename: 图片路径
    :param flags: 1:彩色图片，0:灰度图片 更多详情: https://blog.csdn.net/luxgang/article/details/103509951
    :return: 图片流
    """
    image_flow = cv2.imread(filename=path, flags=flag)

    return image_flow


def show_picture(img_flow: np.ndarray, title: str, wait_time: int = 0):
    """
    显示图片

    :param img_flow: 图片流
    :param title: 标题（同时显示多个, 请不要重名）
    :param wait_time: 等待时间（0:不等待, 一直显示图片）
    :return:
    """
    assert title is not None, "标题必须存在"

    """
    cv2.imshow 显示图片
    :param winname: 标题
    :param mat: 图片流
    """
    cv2.imshow(winname=title, mat=img_flow)

    """
    cv2.waitKey 等待按键输入
    :param delay: 等待时间
    """
    cv2.waitKey(delay=wait_time)

    """
    cv2.destroyAllWindows 销毁所有窗口
    """
    cv2.destroyAllWindows()


def save_picture(img_flow: np.ndarray, path: str):
    """
    将图片流写入文件

    :param img_flow: 图片流
    :param path: 文件路径
    """
    assert path is not None, "图片保存路径不能为空"

    """
    cv2.imwrite 写入图片
    :param filename: 文件路径
    :param images: 图片流
    """
    cv2.imwrite(filename=path, img=img_flow)


def write_text_on_picture(img_flow: np.ndarray, text: str, position: tuple = (0, 0)) -> np.ndarray:
    """
    将文字写到图片上

    :param img_flow: 图片流
    :param text: 文字
    :param position: 文字位置（默认在左上角）
    :return: 图片流
    """
    assert img_flow is not None, "图片流不能为空"

    """
    cv2.putText 写入文字
    :param images: 图片流
    :param text: 文字
    :param org: 坐标
    :param fontFace: 字体
    :param fontScale: 字体大小
    :param color: 颜色
    :param thickness: 粗细
    :param lineType: 线型
    :param bottomLeftOrigin: 是否从左下角开始
    :return: 图片流
    """
    img_flow = cv2.putText(img=img_flow, text=text, org=position,
                           fontFace=setting.FONT, fontScale=setting.FONT_SIZE, color=setting.FONT_COLOR,
                           thickness=setting.FONT_WEIGHT, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

    return img_flow


def scale_picture(gray_img: np.ndarray, scale: float | tuple) -> np.ndarray:
    """
    缩放图片

    :param gray_img: 图片流
    :param scale: 缩放倍数
    :return: 图片流
    """
    assert gray_img is not None, "图片流不能为空"

    """
    cv2.resize 缩放图片
    :param src: 图片流
    :param dsize: 缩放后的尺寸
    :param fx: 水平缩放比例
    :param fy: 垂直缩放比例
    :param interpolation: 插值方式
    :return: 图片流
    """
    if isinstance(scale, float):
        assert scale > 0, "缩放倍数必须大于0"
        gray_img = cv2.resize(src=gray_img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    if isinstance(scale, tuple):
        gray_img = cv2.resize(src=gray_img, dsize=scale, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

    return gray_img


def binarization(image_flow: np.ndarray) -> np.ndarray:
    """
    二值化图像的增强

    :param image_flow: 输入图
    :return: 二值化后的图
    """
    assert image_flow is not None, "图片流不能为空"

    # https://blog.csdn.net/bugang4663/article/details/109589177
    """
    cv2.adaptiveThreshold 自适应阈值二值化
    :param src: 图片流
    :param maxValue: 阈值
    :param adaptiveMethod: 阈值计算方式
    :param thresholdType: 阈值类型
    :param blockSize: 要分成的区域大小 (奇数)
    :param C: 每个区域计算出的阈值的基础上在减去这个常数作为这个区域的最终阈值
    :return: 图片流
    """
    dst = cv2.adaptiveThreshold(src=image_flow, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                thresholdType=cv2.THRESH_BINARY, blockSize=15, C=1)

    return dst
