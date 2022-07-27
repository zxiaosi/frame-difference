#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# @Author : zxiaosi
# @Time : 2022/7/26 14:37
# @desc : 配置文件
import cv2
from pydantic import BaseSettings


class Settings(BaseSettings):
    FIRST_PIC_INPUT: str = "./images/1.png"  # 输入图片
    SECOND_PIC_INPUT: str = "./images/2.png"  # 输入图片
    OUTPUT: str = "./images/result.png"  # 输出结果图片

    ZOOM_RATIO: float | tuple[int, int] = (800, 600)  # 缩放倍数 (0.5 or (400, 300))
    GRAD_MAP_CHOICE: int = 1  # 梯度图选择 (可选值 1, 2, 3)
    DIFF_THRESH: float = 0.5  # 差值阈值 (根据图片自定义)

    FONT = cv2.FONT_HERSHEY_COMPLEX  # 字体
    FONT_SIZE: int = 2  # 字体大小
    FONT_WEIGHT: int = 2  # 字体粗细
    FONT_COLOR: tuple = (255, 255, 255)  # 字体颜色

    class Config:
        case_sensitive = True  # 区分大小写


setting = Settings()
