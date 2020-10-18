import numpy as np
import cv2 as cv

image = cv.imread('pictures/hisColor.jpg')
image = cv.resize(image, (300, 270))
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
'''
直方图均衡化：拉伸 使像素分布更广
    特别是黑色图像 细节都被压缩到直方图的暗端 
    将暗端拉伸的话可以生成一个更均匀分布的直方图，图像会更加清晰
'''
eh_image = cv.equalizeHist(image)
'''
直方图均衡化没考虑到局部图像区域
自适应直方图均衡化是在均衡化过程中只利用局部区域窗口内的直方图分布来构建映射函数
'''
ahe = cv.createCLAHE(clipLimit=0.0, tileGridSize=(8, 8))
ahe_image = ahe.apply(image)
'''
相对自适应有两个改进：
    1.设置阈值 超过部分裁剪并分配 可以避免过度增强噪声点
    2.双线性插值 考虑附近的四个窗口 
'''
clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)

cv.imshow("Histogram Equalization",
          np.hstack([image, eh_image, ahe_image, clahe_image]))
cv.waitKey(0)
cv.destroyAllWindows()