import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.special import comb
import image_processing
import image_processing1
import Square
import Length
# import Bezier
flag=0
#**********************************框选*******************************
def on_mouse(event, x, y, flags, param):
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
        if point1 != point2:
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])
            g_rect = [min_x, min_y, width, height]
            cut_img = img[min_y:min_y + height, min_x:min_x + width]
            cv2.imshow('ROI', cut_img)


def get_image_roi(rgb_image):
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
        cv2.setMouseCallback('image', on_mouse)
        # cv2.startWindowThread()  # 加在这个位置
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
        if key == 13 or key == 32:  # 按空格和回车键退出
            break
    cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return g_rect
#*****************************框选***********************************************

#********************************************比例尺*********************************************
def select_user_roi(image_path):
    '''
    由于原图的分辨率较大，这里缩小后获取ROI，返回时需要重新scale对应原图
    :param image_path:
    :return:
    '''
    orig_image = image_processing.read_image(image_path)
    orig_shape = np.shape(orig_image)
    resize_image = image_processing.resize_image(orig_image, resize_height=800, resize_width=None)
    re_shape = np.shape(resize_image)
    g_rect = get_image_roi(resize_image)
    orgi_rect = image_processing.scale_rect(g_rect, re_shape, orig_shape)
    roi_image = image_processing.get_rect_image(orig_image, orgi_rect)
    image_processing.cv_show_image("RECT", roi_image)
    # image_processing.show_image_rect("image", orig_image, orgi_rect)
    return orgi_rect

def Jshow(img,title=""):
    plt.figure(figsize=(16,9))
    plt.imshow(img,cmap=plt.cm.gray)
    plt.title(title,fontsize=16)
    plt.show()
def calLength(img,thresh,rect,border,plotting_scaleLength):
    '''
    :param img: 读入的图片
    :param thresh:二值化阈值
    :param rect:比例尺的粗略位置
    :param border:比例尺左右竖条宽度
    :return:成功返回具体数值，失败返回None，失败的时候需要通过Jshow查看，修改thresh,rect,border参数
    '''
    plotting_scalepixelCount = -1
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _, imgthresh = cv.threshold(imgGray, thresh, 255, cv.THRESH_BINARY_INV)
    # Jshow(imgthresh,"origin image in Binary")  # 这里显示出来，确保游标尺完整显示出来
    x,y,w,h = rect[0],rect[1],rect[2],rect[3]
    img1 = imgthresh[y:y + h, x:x + w]  # 截取比例尺的位置信息
    Jshow(img1,"plotting-scale image in Binary")  # 这里显示出来，确保游标尺显示出来
    contours1, _ = cv.findContours(img1, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)  # 输出在该范围内的所有可能的轮廓
    for i in range(len(contours1)):
        cutRect1 = cv.boundingRect(contours1[i])  # x,y,w,h
        if (cutRect1[2] > cutRect1[3]) and (cutRect1[2] > (2 * border)):  # 当前轮廓宽高比是否大于1
            x_, y_, w_, h_ = (cutRect1[0] + border, cutRect1[1], cutRect1[2] - 2 * border, cutRect1[3])  # 减去两边的竖条
            cutImg = img1[y_:y_ + h_, x_: x_ + w_]# 减去两边的竖条
            # Jshow(cutImg, "candidate of plotting-scale")
            contours2, _ = cv.findContours(cutImg, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)  # 二次寻找轮廓
            if len(contours2) == 1:
                newRect = cv.boundingRect(contours2[0])
                if newRect[2] / newRect[3] >= 1.7:
                    print("当前游标尺宽度：", cutRect1[2])
                    Jshow(img1[cutRect1[1]:cutRect1[1] + cutRect1[3], cutRect1[0]:cutRect1[0] + cutRect1[2]],"plotting-scale")
                    plotting_scalepixelCount = cutRect1[2]

    if plotting_scalepixelCount != -1:
        return [ plotting_scalepixelCount, plotting_scaleLength/plotting_scalepixelCount]
#***************************比例尺*****************************************************



#******************************面积**********************************
def get_image_roi1(rgb_image):#提取矩形区域（ROI）函数
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
        if key == 65 :  # 按A后显示xyz处理格式的x通道图像
            flag+1
    cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return g_rect#返回当前选框，刷新当前选框

def select_user_roi1(image_path,bilichi):#提取矩形区域（ROI）图像的函数
    '''
    由于原图的分辨率较大，这里缩小后获取ROI，返回时需要重新scale对应原图
    :param image_path:
    :return:
    '''
    orig_image = image_processing1.read_image(image_path)#读取原图
    orig_shape = np.shape(orig_image)#读取原图尺寸
    resize_image = image_processing1.resize_image(orig_image, resize_height=800, resize_width=None)#让原图的高变800，宽按比例缩放
    re_shape = np.shape(resize_image)#缩小后的原图尺寸
    g_rect = get_image_roi1(resize_image)#用缩小后的原图提取的矩形区域
    orgi_rect = image_processing1.scale_rect(g_rect, re_shape, orig_shape)#缩小后的矩形区域重新scale对应原图的区域
    roi_image = image_processing1.get_rect_image(orig_image, orgi_rect)#获取框选图像
    image_processing1.cv_show_image("RECT", roi_image, flag)#显示矩形区域图像
    image_processing1.xsudian(roi_image,bilichi)#打印显示识别物体的像素点
    image_processing1.show_image_rect("image", orig_image, orgi_rect)#显示矩形
    print("image")
    return orgi_rect#返回选框
#*************************************面积***********************************

#*********************************主函数********************************************************************************
if __name__ == '__main__':
    image_path="picture/1bright.tif"
    img = cv.imread(image_path)
#********************先比例尺********************************************************
    img1 = cv.imread(image_path)
    if type(img1) == None:
        print("图片路径出错")
    else:
        plotting_scaleLength = 1000  # 比例尺长度
        rect = select_user_roi(image_path)
        print("选择的矩形区域：", rect)
        # 每一种图片只选一种情况；根据比例尺位置给图片进行分类
        # 第一种情况
        # thresh = 100                  #二值化阈值
        # border = 2                    #游标尺两边数条宽度，二次判定需要
        # img = cv.imread("./img/1197272753.jpg")

        # 第二种情况
        # border = 5  # 游标尺两边数条宽度，二次判定需要
        # thresh = 50  # 二值化阈值
        # img = cv.imread("./img/20220114-01.jpg")
        # img = cv.imread("./img/20220114-01-metric.jpg")

        # 第三种情况
        thresh = 100  # 二值化阈值
        border = 4  # 游标尺两边数条宽度，二次判定需要
        # # img = cv.imread("./img/2.1.tif")#
        # 第五种...
        data = calLength(img1, thresh, rect, border, plotting_scaleLength)
        if data != None:
            print("比例尺像素个数：{}，单个像素物理长度：{}".format(data[0], data[1]))
        else:
            print("请查验thresh，border，rect参数设置")

#**************比例尺**********************************************************************
    bilichi = int( data[1]*100)/100
# *****************************长度***************************************************
class MyBezier:
    # image_path="picture/1bright.tif"
    def __init__(self):
        self.img = cv2.imread(image_path)  # 生成背景, cv2.IMREAD_GRAYSCALE
        self.xs = list()  # 保存点的x坐标
        self.ys = list()  # 保存点的y坐标
        self.pts = list()  # 存一个点，方便画

    def draw(self, event, x, y, flags, param):  # 绘图
        if event == cv2.EVENT_LBUTTONDOWN:
            self.img = cv2.imread(image_path)  # 不清除的话会保留原有的图, cv2.IMREAD_GRAYSCALE
            self.xs.append(x)
            self.ys.append(y)
            self.pts.append([x, y])
            self.bezier(self.xs, self.ys)  # Bezier曲线
            # 画折线
            tmp = np.array(self.pts, np.int32)
            tmp = [tmp.reshape((-1, 1, 2))]
            cv2.polylines(self.img, tmp, False, (0, 0, 255))

            cv2.imshow("lkz", self.img)  # 重构图

    def bezier(self, XVecs, YVecs):  # Bezier曲线公式转换，获取x和y
        t = np.linspace(0, 1)  # t 范围0到1
        n = len(XVecs) - 1
        v_x, v_y = 0, 0  # 这里的x、y都是一组点向量
        for i, x in enumerate(XVecs):
            # x(t)=Σ(i=0,n)xi*B(i,n)(t)
            # B(i,n)(t)=comb(n,i)*(t^i)*(1-t)^(n-i)
            v_x = v_x + x * pow(t, i) * pow(1 - t, n - i) * comb(n, i)  # comb 组合，perm 排列

        for i, y in enumerate(YVecs):
            v_y = v_y + y * pow(t, i) * pow(1 - t, n - i) * comb(n, i)

        # 画线
        points = []
        for i in range(len(v_x)):
            points.append([v_x[i], v_y[i]])
        tmp = np.array(points, np.int32)
        tmp = [tmp.reshape((-1, 1, 2))]
        cv2.polylines(self.img, tmp, False, (0, 0, 255))
        # 计算长度
        # area_list[] 存储每一微小步长的曲线长度
        area_list = [np.sqrt((v_x[i] - v_x[i - 1]) ** 2 + (v_y[i] - v_y[i - 1]) ** 2) for i in range(1, len(t))]
        area = sum(area_list)  # 求和计算曲线在t:[0,2*pi]的长度
        # 在图像上输出显示生物曲线长度
        length=area*bilichi
        long = "%.4f" % length
        cv2.putText(self.img, long, ((int)(v_x[i]), (int)(v_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1,cv2.LINE_AA)

    def main(self):
        cv2.namedWindow('lkz')
        cv2.setMouseCallback('lkz', self.draw)  # 鼠标按下事件
        while (1):
            cv2.imshow('lkz', self.img)
            code = cv2.waitKey(100)
            if code == ord('q'):  # 按下q退出
                break
        cv2.destroyAllWindows()


# ************************************************************长度********************************************************************
#*************************再体长************************************************************
tt = MyBezier()
tt.main()
#************************************体长**************************************************

#********************************再面积******************************************************
rect = select_user_roi1(image_path,bilichi)  # 调用提取矩形区域（ROI）图像的函数

#**********************清空注销*****************************************
cv2.destroyAllWindows()
cv2.waitKey(0)




