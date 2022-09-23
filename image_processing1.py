import os
import glob
import cv2

import numpy as np
import matplotlib.pyplot as plt


def show_image(title, image):
    '''
    调用matplotlib显示RGB图片
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()


def cv_show_image(title, image,flag):#显示图片函数
    '''
    调用OpenCV显示RGB图片
    :param title: 图像标题
    :param image: 输入RGB图像
    :return:
    '''
    channels = image.shape[-1]
    if channels == 3:
        if flag % 2:#如果不按下A还是显示彩色通道
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
            cv2.imshow(title, image)
        else:#如果按下A则显示XYZ的X通道图片
            xyz = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
            image, y, z = cv2.split(xyz)
            cv2.imshow(title, image)
    cv2.waitKey(0)


def read_image(filename, resize_height=None, resize_width=None, normalization=False):#读取图像路径获取图像函数
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的RGB图片数据
    '''

    bgr_image = cv2.imread(filename)#读取图像路径获取图像
    # bgr_image = cv2.imread(filename,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)
    if bgr_image is None:#如果图像不存在
        print("Warning:不存在:{}", filename)
        return None
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB

    rgb_image = resize_image(rgb_image, resize_height, resize_width)#图像缩小
    rgb_image = np.asanyarray(rgb_image)#传递数据
    if normalization:
        rgb_image = rgb_image / 255.0
    return rgb_image#返回（缩小的)图像


def resize_image(image, resize_height, resize_width):#缩小图像
    '''
    :param image:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    image_shape = np.shape(image)#获取原图尺寸
    height = image_shape[0]#原图的高
    width = image_shape[1]#原图的宽
    if (resize_height is None) and (resize_width is None):  # 如果重定义高宽都不给出，不作任何缩放
        return image
    if resize_height is None:#如果不给出缩放的高数据
        resize_height = int(height * resize_width / width)#则高为原高*缩放的宽/原宽（计算比例）高=原高*缩放比例
    elif resize_width is None:#如果没有给出缩放的宽数据
        resize_width = int(width * resize_height / height)#则宽为原宽*缩放的高/原高（计算比例）宽=原宽*缩放比例
    image = cv2.resize(image, dsize=(resize_width, resize_height))#将原图缩放成给定的宽高的样子
    #resize是opencv库中的一个函数，主要起到对图片进行缩放的作用。
    #以下代码就可以将原图片转化为宽和长分别为300，300的图片。width和height可以自己任意指定，不论大小。
    #img = cv.resize(img,(width,height))
    return image#返回缩放的图像


def scale_image(image, scale):#将缩放的图像恢复大小
    '''
    :param image:
    :param scale: (scale_w,scale_h)
    :return:
    '''
    image = cv2.resize(image, dsize=None, fx=scale[0], fy=scale[1])#将缩放的图像恢复大小
    return image
    #resize(InputArray src, OutputArray dst, Size dsize,double fx=0, double fy=0, int interpolation=INTER_LINEAR )
    #InputArray src ：输入，原图像，即待改变大小的图像；
    # OutputArray dst： 输出，改变后的图像。这个图像和原图像具有相同的内容，只是大小和原图像不一样而已；
    # dsize：输出图像的大小，如上面例子（300，300）。
    # 其中，fx和fy就是下面要说的两个参数，是图像width方向和height方向的缩放比例。
    # fx：width方向的缩放比例
    # fy：height方向的缩放比例



def get_rect_image(image, rect):#获取矩形框选的图像
    '''
    :param image:
    :param rect: [x,y,w,h]
    :return:
    '''
    x, y, w, h = rect
    cut_img = image[y:(y + h), x:(x + w)]
    return cut_img


def scale_rect(orig_rect, orig_shape, dest_shape):#对矩形缩放回去
    '''
    对图像进行缩放时，对应的rectangle也要进行缩放
    :param orig_rect: 原始图像的rect=[x,y,w,h]
    :param orig_shape: 原始图像的维度shape=[h,w]
    :param dest_shape: 缩放后图像的维度shape=[h,w]
    :return: 经过缩放后的rectangle
    '''
    new_x = int(orig_rect[0] * dest_shape[1] / orig_shape[1])
    new_y = int(orig_rect[1] * dest_shape[0] / orig_shape[0])
    new_w = int(orig_rect[2] * dest_shape[1] / orig_shape[1])
    new_h = int(orig_rect[3] * dest_shape[0] / orig_shape[0])
    dest_rect = [new_x, new_y, new_w, new_h]
    return dest_rect#返回缩放回去的矩形


def show_image_rect(win_name, image, rect):#显示矩形框
    '''
    :param win_name:
    :param image:
    :param rect:
    :return:
    '''
    x, y, w, h = rect
    point1 = (x, y)
    point2 = (x + w, y + h)
    cv2.rectangle(image, point1, point2, (0, 0, 255), thickness=2)
    cv_show_image(win_name, image)


def rgb_to_gray(image):#转灰度图
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def save_image(image_path, rgb_image, toUINT8=True):#保存图像
    if toUINT8:
        rgb_image = np.asanyarray(rgb_image * 255, dtype=np.uint8)
    if len(rgb_image.shape) == 2:  # 若是灰度图则转为三通道
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, bgr_image)


def combime_save_image(orig_image, dest_image, out_dir, name, prefix):#保存图像
    '''
    命名标准：out_dir/name_prefix.jpg
    :param orig_image:
    :param dest_image:
    :param image_path:
    :param out_dir:
    :param prefix:
    :return:
    '''
    dest_path = os.path.join(out_dir, name + "_" + prefix + ".jpg")
    save_image(dest_path, dest_image)

    dest_image = np.hstack((orig_image, dest_image))
    save_image(os.path.join(out_dir, "{}_src_{}.jpg".format(name, prefix)), dest_image)

def xsudian(image,bilichi):#grabcut函数分割图像并计算前景（白色）像素点
    img = image

    shape = img.shape
    row_count = shape[0]
    col_count = shape[1]

    # Step2. 创建掩模、背景图和前景图
    mask = np.zeros(img.shape[:2], np.uint8)  # 创建大小相同的掩模
    bgdModel = np.zeros((1, 65), np.float64)  # 创建背景图像
    fgdModel = np.zeros((1, 65), np.float64)  # 创建前景图像

    # Step3. 初始化矩形区域
    # 这个矩形必须完全包含前景
    rect = (0, 0, shape[0], shape[1])  # 格式为（x, y, w, h）

    # Step4. GrubCut算法，迭代5次
    # mask的取值为0,1,2,3
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)  # 迭代5次

    # Step5. mask中，值为2和0的统一转化为0, 1和3转化为1

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img1 = img * mask2[:, :, np.newaxis]  # np.newaxis 插入一个新维度，相当于将二维矩阵扩充为三维

    # mask3 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    # img2 = img * mask3[:,:,np.newaxis] # np.newaxis 插入一个新维度，相当于将二维矩阵扩充为三维
    # Get the background
    background = img - img1

    # Change all pixels in the background that are not black to red 背景变红
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [0, 0, 255]

    # Add the background and the image 最后的图像是背景+识别图像
    final = background + img1

    pic = final.copy()
    shape = final.shape
    row_count = shape[0]
    col_count = shape[1]

    p = 0
    for row in range(0, row_count):
        for col in range(0, col_count):
            a = pic[row, col]
            r = a[2]
            g = a[1]
            b = a[0]
            if (r != 255 or g != 0 or b != 0):#最后的图像里面不是红的像素都换成白的（即把识别出来的物体变白）
                pic[row, col] = [255, 255, 255]  # 颜色，RGB前景换白
                p = p + 1#像素点++
                # print(p)

    for row in range(0, row_count):
        for col in range(0, col_count):
            a = pic[row, col]
            r = a[2]
            g = a[1]
            b = a[0]
            if (r == 255 and g == 0 and b == 0):#把背景变黑
                pic[row, col] = [0, 0, 0]  # 颜色，RGB后景换黑
    # cv2.imshow("bgd_white", final)
    square = p * bilichi * bilichi
    s="%.4f" % square
    # cv2.putText(pic, s, (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1,cv2.LINE_AA)
    while (1):
        cv2.putText(pic, s, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("fgd_white", pic)  # 输出最后前景白背景黑的图片
        code = cv2.waitKey(100)
        if code == ord('q'):  # 按下q退出
            break

    print("像素点个数:")#输出像素点个数
    print(p)
    print("生物面积:")#输出像素点个数
    print(square)

    # return p
    # cv2.imshow("img_cut", img1)
    cv2.waitKey(0)


if __name__ == "__main__":
    image_path = "../dataset/test_images/src.jpg"
    image = read_image(image_path, resize_height=None, resize_width=None)
    image = rgb_to_gray(image)
    orig_shape = np.shape(image)  # shape=(h,w)
    orig_rect = [50, 100, 100, 200]  # x,y,w,h
    print("orig_shape:{}".format(orig_shape))
    show_image_rect("orig", image, orig_rect)

    dest_image = resize_image(image, resize_height=None, resize_width=200)
    dest_shape = np.shape(dest_image)
    print("dest_shape:{}".format(dest_shape))
    dest_rect = scale_rect(orig_rect, orig_shape, dest_shape)
    show_image_rect("dest", dest_image, dest_rect)


