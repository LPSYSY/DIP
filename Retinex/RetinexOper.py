import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
'''
映射到人眼中的图像和光的
    长波(R)、
    中波(G)、
    短波(B)、
    以及物体的反射性质有关
I(x,y)=R(x,y)∙L(x,y) 
I(x,y)代表观察到的图像
R(x,y)代表物体的反射属性
L(x,y)代表物体表面的光照
'''


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


def singleScaleRetinexProcess(img, sigma):
    '''
    假定是光照I(x,y)是缓慢变化的 也就是低频的
    要从I(x,y)中还原出R(x,y)
    所以可以通过低通滤波器得到光照分量

    对源公式两边同时取对数 得到log(R)=log(I)−log(L)​
    通过高斯模糊低通滤波器对原图进行滤波 公式为L=F∗I
    得到L后，利用第一步的公式得到log(R) 然后通过exp函数得到R
    '''
    # 高斯内核的大小为(0,0) 则会根据sigma自行计算
    temp = cv.GaussianBlur(img, (0, 0), sigma)
    # temp==0用0.01代替 否则保持原值temp
    gaussian = np.where(temp == 0, 0.01, temp)
    # 取对数 避免出现log10(0) 相减
    retinex = np.log10(img + 0.01) - np.log10(gaussian)
    return retinex


def multiScaleRetinexProcess(img, sigma_list):
    '''
    多尺度
    单尺度处理之后相加求平均即可
    '''
    retinex = np.zeros_like(img * 1.0)
    for sigma in sigma_list:
        retinex = singleScaleRetinexProcess(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex


def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration


def multiScaleRetinexWithColorRestorationProcess(img, sigma_list, G, b, alpha,
                                                 beta):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinexProcess(img, sigma_list)
    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)
    return img_msrcr


def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img


# 量化图像
def touint8(img):
    for i in range(img.shape[2]):
        img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / \
                       (np.max(img[:, :, i]) - np.min(img[:, :, i])) * 255
    img = np.uint8(np.minimum(np.maximum(img, 0), 255))
    return img


# SSR
def SSR(img, sigma=300):
    ssr = singleScaleRetinexProcess(img, sigma)
    ssr = touint8(ssr)
    return ssr


# MSR
def MSR(img, sigma_list=[15, 80, 250]):
    msr = multiScaleRetinexProcess(img, sigma_list)
    msr = touint8(msr)
    return msr


# MSRCR
def MSRCR(img,
          sigma_list=[15, 80, 250],
          G=5,
          b=25,
          alpha=125,
          beta=46,
          low_clip=0.01,
          high_clip=0.99):
    msrcr = multiScaleRetinexWithColorRestorationProcess(
        img, sigma_list, G, b, alpha, beta)
    msrcr = touint8(msrcr)
    msrcr = simplestColorBalance(msrcr, low_clip, high_clip)
    return msrcr


if __name__ == "__main__":
    image = cv.resize(cv.imread('pictures/cloud.jpg'), (300, 300))

    ssr = SSR(image)
    msr = MSR(image)
    msrcr = MSRCR(image)

    # cv_show('res', showTwoPics(image, ssr))
    # cv_show('res', showTwoPics(image, msr))
    # cv_show('res', showTwoPics(image, msrcr))

    plt.subplot(1, 4, 1), plt.imshow(image), plt.title("origin"), plt.xticks(
        []), plt.yticks([])
    plt.subplot(1, 4, 2), plt.imshow(ssr), plt.title("ssr"), plt.xticks(
        []), plt.yticks([])
    plt.subplot(1, 4, 3), plt.imshow(msr), plt.title("msr"), plt.xticks(
        []), plt.yticks([])
    plt.subplot(1, 4, 4), plt.imshow(msrcr), plt.title("msrcr"), plt.xticks(
        []), plt.yticks([])
    plt.show()