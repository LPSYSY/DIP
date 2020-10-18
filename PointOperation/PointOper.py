import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
import random


def cv_show(name, img):
    '''
    显示图像
    '''
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def showTwoPics(imgOrigin, imgResult):
    '''
    同时显示两张图片进行对比
    '''
    return np.hstack((imgOrigin, imgResult))


def sp_noise(img, prob=0.005):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # [0,1)
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output


def ImageNegative(img):
    '''
    图像反转
    应用: 显示医学图像和单色正片拍摄屏幕
    转换方程: T： G(x, y) = L - F(x, y) (L是最大强度值)
    '''
    L = 255
    imgOrigin = img
    # cv_show('origin', imgOrigin)
    imgNeg = L - imgOrigin
    # cv_show('imgResult', imgNeg)
    cv_show('res', showTwoPics(imgOrigin, imgNeg))


def ContrastStretch(img):
    '''
    灰度拉伸
    定义：灰度拉伸，也称对比度拉伸，是一种简单的线性点运算。作用：扩展图像的直方图，使其充满整个灰度等级范围内
    公式：g(x,y) = 255 / (B - A) * [f(x,y) - A],
    其中，A = min[f(x,y)],最小灰度级；B = max[f(x,y)],最大灰度级；f(x,y)为输入图像,g(x,y)为输出图像
    缺点：如果灰度图像中最小值A=0，最大值B=255，则图像没有什么改变
    '''
    imgOrigin = img
    rows, cols = imgOrigin.shape
    flatGray = imgOrigin.reshape((cols * rows)).tolist()
    A = min(flatGray)
    B = max(flatGray)
    imgResult = np.uint8(255 / (B - A) * (imgOrigin - A) + 0.5)
    cv_show('res', showTwoPics(imgOrigin, imgResult))


def CompressionDynamicRange(img):
    '''
    动态范围压缩
    若图像处理后的动态范围超过了设备显示的范围，需要对图像进行动态范围压缩
    转换公式：s = c*log(1+|r|)  c是度量常数 r是当前灰度 s是转换后灰度
    若将图像灰度范围[0, 255]压缩到[0, 150] 则：c = 150 / log(1+255)
    '''
    imgOrigin = img
    c = 150 / math.log(1 + 255)
    rows, cols = imgOrigin.shape
    imgResult = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            imgResult[i, j] = c * math.log(1 + imgOrigin[i, j])
    cv_show('res', showTwoPics(imgOrigin, imgResult))


def GreyLevelSlice(img):
    '''
    灰度切片
    灰度切片既可以强调一组灰度值而减少其他所有灰度值
    也可以强调一组灰度值而不考虑其他灰度值
    '''
    imgOrigin = img
    rows, cols = imgOrigin.shape
    imgResult = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if imgOrigin[i, j] > 100 and imgOrigin[i, j] < 180:
                imgResult[i, j] = 150
            else:
                imgResult[i, j] = 25
    print(imgResult)
    cv_show('res', showTwoPics(imgOrigin, imgResult))


def ImageSub(img, template):
    '''
    图像相减
    调平图像的不均匀部分和检测两幅图像之间的变化
    应用: 从场景中减去背景光照的变化，更容易会分析出情景对象
    想将文本从背景页面中分离出来，但是不能调整光照，显微镜通常就这样，从原始图像减去光场图像，消除背景强度的变化
    '''
    imgOrigin = img
    imgResult = img - template
    cv_show('res', showTwoPics(imgOrigin, imgResult))


def ImageAverage(img):
    '''
    图像平均
    多个图像的平均值，应用于图像去噪
    '''
    imgOrigin = img
    imgNoise1 = sp_noise(img)
    imgNoise2 = sp_noise(img)
    imgNoise3 = sp_noise(img)
    imgNoise4 = sp_noise(img)
    imgNoise5 = sp_noise(img)
    rows, cols = imgOrigin.shape
    imgResult = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            imgResult[i,
                      j] = int(0.2 * imgNoise1[i, j] + 0.2 * imgNoise2[i, j] +
                               0.2 * imgNoise3[i, j] + 0.2 * imgNoise4[i, j] +
                               0.2 * imgNoise5[i, j])

    plt.subplot(2, 4, 1), plt.imshow(imgOrigin, 'gray'), plt.title("origin")
    plt.subplot(2, 4, 2), plt.imshow(imgNoise1, 'gray'), plt.title("noise1")
    plt.subplot(2, 4, 3), plt.imshow(imgNoise2, 'gray'), plt.title("noise2")
    plt.subplot(2, 4, 4), plt.imshow(imgNoise3, 'gray'), plt.title("noise3")
    plt.subplot(2, 4, 5), plt.imshow(imgNoise4, 'gray'), plt.title("noise4")
    plt.subplot(2, 4, 6), plt.imshow(imgNoise4, 'gray'), plt.title("noise5")
    plt.subplot(2, 4, 7), plt.imshow(imgResult, 'gray'), plt.title("result")
    plt.show()


if __name__ == "__main__":
    # imgOrigin = cv.imread('pictures/medicine.png', 0)
    # ImageNegative(imgOrigin)

    # imgOrigin = cv.imread('pictures/flower.jpg', 0)
    # ContrastStretch(imgOrigin)

    # imgOrigin = cv.imread('pictures/circle.jpg', 0)
    # CompressionDynamicRange(imgOrigin)

    # imgOrigin = cv.imread('pictures/lena.jpg', 0)
    # GreyLevelSlice(imgOrigin)

    # imgOrigin = cv.imread('pictures/paper.jpg', 0)
    # imgTmp = cv.imread('pictures/template.jpg', 0)
    # ImageSub(imgOrigin, imgTmp)

    imgOrigin = cv.imread('pictures/cat.jpg', 0)
    ImageAverage(imgOrigin)

    pass