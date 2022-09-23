import numpy as np
from scipy.special import comb
import cv2
#
# pic_dir = 'picture'
# pic_name = '2light.tif'  # 骨科图片
# img_file = pic_dir+"/"+pic_name

class MyBezier:
    def __init__(self):
        self.img = cv2.imread(img_file)  # 生成背景, cv2.IMREAD_GRAYSCALE
        self.xs = list()  # 保存点的x坐标
        self.ys = list()  # 保存点的y坐标
        self.pts = list()  # 存一个点，方便画

    def draw(self, event, x, y, flags, param):  # 绘图
        if event == cv2.EVENT_LBUTTONDOWN:
            self.img = cv2.imread(img_file)  # 不清除的话会保留原有的图, cv2.IMREAD_GRAYSCALE
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
        #在图像上输出显示生物曲线长度
        long = "%.4f" % area
        cv2.putText(self.img, long, ((int)(v_x[i]), (int)(v_y[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)


    def main(self):
        cv2.namedWindow('lkz')
        cv2.setMouseCallback('lkz', self.draw)  # 鼠标按下事件
        while (1):
            cv2.imshow('lkz', self.img)
            code = cv2.waitKey(100)
            if code == ord('q'):  # 按下q退出
                break
        cv2.destroyAllWindows()
#
tt = MyBezier()
tt.main()
