# Image-Processing-Insect
计算机图像处理/软件工程A课程的课设|昆虫显微图像的生物量测量系统
# 项目介绍
## 开发背景
  
生物量通常指一个种群个体的总数量或总重量，是表征生物生命活力的指标。  
研究小型土壤节肢动物生物量的意义在于，有效检测在土壤环境下这类小型动物生活状况，测定其在生态位上对应营养级的生产力。  
同时，这也是研究生态系统物质与能量流动的基本方法。  
    
目前对于小型土壤节肢动物的生物量传统测量主要是通过显微镜下形态观察、统计各类群个体数，手动测量体长或是拍照记录虫体投影面积（模型估算法），计算生物量。  
国外对于微土壤动物，如线虫的生物量测算，结合了显微镜和数字图像分析仪，同时在近期使用了代谢足迹和大小光谱等新方法。
## 项目功能
本系统大致有以下几点设计目标：
1.	导入昆虫显微图片，用户输入比例尺单位、数字
2.	框选比例尺图像，再进行图像灰度化或二值化处理，并调用opencv库函数，由系统识别计算比例尺（图上一像素点对应的实际长度）
3.	用Bezier等函数算法进行体长曲线的打点和优化
4.	框选待识别投影面积的虫体，用grabcut等函数算法识别绘制出虫体轮廓，并计算生物面积
5.	在图像上输出对应的结果（如体长曲线打点完毕后能实时在图像上输出体长数值）
## 开发环境
开发语言为Python3.7.0  
开发工具为PyCharm 2020.3.4 x64， 采用了opencv-python库，版本为4.5.5.62  
开发系统为Windows10 
# 运行环境搭建
（本系统尚未做可执行.exe文件）  
本系统能够直接将源码文件夹拖入pycharm中运行，但事先需要install的头文件大致有：  
①	OpenCV库(需要用到各类cv.或cv2.函数，请自行配置好电脑环境)  
②	cv2  
③	numpy  
④	matplotlib  
⑤	glob  
⑥	os  
⑦	scipy.special  
⑧	comb  
以上的包或者环境配置网上随便一搜都有教程，基本上都只需要在cmd里install xxxx一下就行了，其实很多pycharm也都自带了，请读者自行下载。  
# 源码的说明
Square.py为框选和测量面积模块的主文件  
image_processing1.py为实现框选功能和分割图像函数而调用的文件  
Length.py为框选和提取比例尺模块的主文件  
image_processing.py为实现框选功能而调用的文件  
Bezier.py为生物曲线拟合和长度测量模块主文件  
Manage为整合三个模块，实现系统功能的主文件  
Picture为图片数据集  
# 最后的效果  
![image](https://user-images.githubusercontent.com/66285048/191899909-848fef7f-8d87-4e07-9779-37557385672e.png)  
![image](https://user-images.githubusercontent.com/66285048/191899954-134e5100-fa6b-4be4-9ea4-3ac2da43c6cc.png)  
![image](https://user-images.githubusercontent.com/66285048/191899974-e77320cb-d885-49b2-9ced-d5bd7b82c3dc.png)  
![image](https://user-images.githubusercontent.com/66285048/191899995-ea5263e0-6581-4a97-9365-ecebb72196e5.png)  



