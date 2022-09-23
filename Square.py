# -*-coding: utf-8 -*-
"""
    @Project: IntelligentManufacture
    @File   : user_interaction.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-02-21 15:03:18
"""
# -*- coding: utf-8 -*-

import cv2
import image_processing1#引入图像框选py文件
import numpy as np
# import cut

global img
global point1, point2
global g_rect#全局变量，存储框选坐标信息
flag=0

def on_mouse(event, x, y, flags, param):#鼠标画框函数
    global img, point1, point2, g_rect
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
        print("1-EVENT_LBUTTONDOWN")
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
        print("2-EVENT_FLAG_LBUTTON")
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), thickness=2)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
        print("3-EVENT_LBUTTONUP")
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), thickness=2)
        cv2.imshow('image', img2)
        if point1 != point2:#起始点不一样时，得出框选的各个数值
            min_x = min(point1[0], point2[0])#左上角的x
            min_y = min(point1[1], point2[1])#左上角的y
            width = abs(point1[0] - point2[0])#宽
            height = abs(point1[1] - point2[1])#高
            g_rect = [min_x, min_y, width, height]#存储数据
            cut_img = img[min_y:min_y + height, min_x:min_x + width]#剪切图片
            cv2.imshow('ROI', cut_img)#显示框选区域


def get_image_roi(rgb_image):#提取矩形区域（ROI）函数
    '''
    获得用户ROI区域的rect=[x,y,w,h]
    :param rgb_image:
    :return:
    '''
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    global img
    img = bgr_image
    cv2.namedWindow('image')
    while True:
        cv2.setMouseCallback('image', on_mouse)#刷新当前选框
        #鼠标回调函数 on_mouse能传回当前发生的鼠标事件类型和x,y值等（↑是调用上面的on_mouse函数）
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
        if key == 32 or key==13:  # 按回车键退出
            break
        # if key == 65 :  # 按A后显示xyz处理格式的x通道图像
        #     flag+1
    cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return g_rect#返回当前选框，刷新当前选框



def select_user_roi(image_path):#提取矩形区域（ROI）图像的函数
    '''
    由于原图的分辨率较大，这里缩小后获取ROI，返回时需要重新scale对应原图
    :param image_path:
    :return:
    '''
    orig_image = image_processing1.read_image(image_path)#读取原图
    orig_shape = np.shape(orig_image)#读取原图尺寸
    resize_image = image_processing1.resize_image(orig_image, resize_height=800, resize_width=None)#让原图的高变800，宽按比例缩放
    re_shape = np.shape(resize_image)#缩小后的原图尺寸
    g_rect = get_image_roi(resize_image)#用缩小后的原图提取的矩形区域
    orgi_rect = image_processing1.scale_rect(g_rect, re_shape, orig_shape)#缩小后的矩形区域重新scale对应原图的区域
    roi_image = image_processing1.get_rect_image(orig_image, orgi_rect)#获取框选图像
    image_processing1.cv_show_image("RECT", roi_image, flag)#显示矩形区域图像, flag
    image_processing1.xsudian(roi_image)#打印显示识别物体的像素点
    # p=image_processing1.MJ(roi_image)
    image_processing1.show_image_rect("image", orig_image, orgi_rect)#显示矩形
    print("image")
    # print("面积像素个数")
    # print(p)
    return orgi_rect#返回选框



if __name__ == '__main__':
    image_path = "../picture/2light.tif"
    rect = select_user_roi(image_path)#调用提取矩形区域（ROI）图像的函数
    # print("面积像素个数：{}，生物面积：{}",rect,rect*6)
