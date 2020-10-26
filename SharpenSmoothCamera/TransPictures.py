import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math


def cv_show(name, img):
    '''
    显示图像
    '''
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def showTwoPicts(imgOrigin, imgEdge):
    '''
    同时显示原图、边界
    '''
    cv_show('res', np.hstack((imgOrigin, imgEdge)))


def showThreePicts(imgOrigin, img, imgFilter):
    '''
    同时显示原图、椒盐、平滑
    '''
    cv_show('res', np.hstack((imgOrigin, img, imgFilter)))


def salt_pepper(image, salt, pepper):
    """
    添加椒盐噪声的图像
    :param image: 输入图像
    :param salt: 盐比例
    :param pepper: 椒比例
    :return: 添加了椒盐噪声的图像
    """
    height = image.shape[0]
    width = image.shape[1]
    pertotal = salt + pepper  #总噪声占比
    noise_image = image.copy()
    noise_num = int(pertotal * height * width)
    for i in range(noise_num):
        rows = np.random.randint(0, height - 1)
        cols = np.random.randint(0, width - 1)
        if (np.random.randint(0, 100) < salt * 100):
            noise_image[rows][cols] = 255
        else:
            noise_image[rows][cols] = 0
    return noise_image


def boxFilter(img):
    '''
    方形滤波
    -1代表原图深度
    '''
    # imgFilterBox1 = cv.boxFilter(img, -1, (5, 5))

    kernel = np.array(([0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125
                                                 ], [0.0625, 0.125, 0.0625]),
                      dtype="float32")

    imgFilterBox2 = cv.filter2D(img, -1, kernel)
    return imgFilterBox2


def averFilter(img):
    '''
    均值滤波
    '''
    imgFilterAver = cv.blur(img, (3, 3))
    return imgFilterAver


def medianFilter(img):
    '''
    中值滤波
    '''
    imgFilterMed = cv.medianBlur(img, 5)
    return imgFilterMed


def GuassFilter(img):
    '''
    高斯滤波
    '''
    imgFilterGau = cv.GaussianBlur(img, (5, 5), 0)
    return imgFilterGau


def TwoEdgesFilter(img):
    '''
    双边滤波
    sigmaColor：颜色空间的标准方差 一般尽可能大
    sigmaSpace：坐标空间的标准方差(像素单位) 一般尽可能小
    '''
    imgFilterBila = cv.bilateralFilter(img, 5, 31, 31)
    return imgFilterBila


def SobelSharpen(img):
    '''
    Sobel算子进行图像锐化
    '''
    imgEdgeX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    imgEdgeY = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    imgEdgeX = cv.convertScaleAbs(imgEdgeX)
    imgEdgeY = cv.convertScaleAbs(imgEdgeY)
    imgEdgeSob = cv.addWeighted(imgEdgeX, 0.5, imgEdgeY, 0.5, 0)
    return imgEdgeSob


def LaplaceSharpen(img):
    '''
    Laplace算子进行图像锐化
    '''
    imgEdge = cv.Laplacian(img, cv.CV_64F)
    imgEdgeLap = cv.convertScaleAbs(imgEdge)
    return imgEdgeLap


def ScharrSharpen(img):
    '''
    Scharr算子进行图像锐化
    '''
    imgEdgeX = cv.Scharr(img, cv.CV_64F, 1, 0)
    imgEdgeY = cv.Scharr(img, cv.CV_64F, 0, 1)
    imgEdgeX = cv.convertScaleAbs(imgEdgeX)
    imgEdgeY = cv.convertScaleAbs(imgEdgeY)
    imgEdgeScharr = cv.addWeighted(imgEdgeX, 0.5, imgEdgeY, 0.5, 0)
    return imgEdgeScharr


def CannySharpen(img):
    '''
    Canny边缘检测
    '''
    imgEdgeCan = cv.Canny(img, 50, 100)
    return imgEdgeCan
