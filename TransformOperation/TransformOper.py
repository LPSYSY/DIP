import cv2 as cv
import numpy as np
import random
import math
import matplotlib.pyplot as plt
'''
在频域进行操作
G(u,v)=F(u,v)*H(u,v)
F(u,v) 是原图经过快速傅里叶变换转换到频域的频谱
H(u,v)是在频域执行的操作
G(u,v)是在频域处理后的频谱结果
最后G(u,v)可以通过快速傅里叶反变换(ifft)得到滤波的图像
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


def showThreePics(imgOrigin, imgMiddle, imgResult):
    '''
    同时显示三张图片进行对比
    '''
    return np.hstack((imgOrigin, imgMiddle, imgResult))


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


def low_pass_filtering(image, radius):
    """
    低通滤波函数
    :param image: 输入图像
    :param radius: 半径
    :return: 滤波结果
    """
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift = np.fft.fftshift(fft)

    # 得到中心像素
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)

    # 构建掩模，256位，两个通道
    mask = np.zeros((rows, cols, 2), np.float32)
    mask[mid_row - radius:mid_row + radius,
         mid_col - radius:mid_col + radius] = 1

    # 给傅里叶变换结果乘掩模
    fft_filtering = dshift * mask
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv.idft(ishift)
    image_filtering = cv.magnitude(image_filtering[:, :, 0],
                                   image_filtering[:, :, 1])
    # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
    cv.normalize(image_filtering, image_filtering, 0, 1, cv.NORM_MINMAX)
    return image_filtering


def high_pass_filtering(image, radius, n):
    """
    高通滤波函数
    :param image: 输入图像
    :param radius: 半径
    :param n: ButterWorth滤波器阶数
    :return: 滤波结果
    """
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift = np.fft.fftshift(fft)

    # 得到中心像素
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)

    # 构建ButterWorth高通滤波掩模

    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
            # 计算(i, j)到中心点的距离
            d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
            try:
                mask[i, j, 0] = mask[i, j,
                                     1] = 1 / (1 + pow(radius / d, 2 * n))
            except ZeroDivisionError:
                mask[i, j, 0] = mask[i, j, 1] = 0
    # 给傅里叶变换结果乘掩模
    fft_filtering = dshift * mask
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv.idft(ishift)
    image_filtering = cv.magnitude(image_filtering[:, :, 0],
                                   image_filtering[:, :, 1])
    # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
    cv.normalize(image_filtering, image_filtering, 0, 1, cv.NORM_MINMAX)
    return image_filtering


def bandpass_filter(image, radius, w, n=1):
    """
    带通滤波函数
    :param image: 输入图像
    :param radius: 带中心到频率平面原点的距离
    :param w: 带宽
    :param n: 阶数
    :return: 滤波结果
    """
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift = np.fft.fftshift(fft)

    # 得到中心像素
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)

    # 构建掩模，256位，两个通道
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
            # 计算(i, j)到中心点的距离
            d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
            if radius - w / 2 < d < radius + w / 2:
                mask[i, j, 0] = mask[i, j, 1] = 1
            else:
                mask[i, j, 0] = mask[i, j, 1] = 0

    # 给傅里叶变换结果乘掩模
    fft_filtering = dshift * np.float32(mask)
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv.idft(ishift)
    image_filtering = cv.magnitude(image_filtering[:, :, 0],
                                   image_filtering[:, :, 1])
    # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
    cv.normalize(image_filtering, image_filtering, 0, 1, cv.NORM_MINMAX)
    return image_filtering


def bandstop_filter(image, radius, w, n=1):
    """
    带阻滤波函数
    :param image: 输入图像
    :param radius: 带中心到频率平面原点的距离
    :param w: 带宽
    :param n: 阶数
    :return: 滤波结果
    """
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift = np.fft.fftshift(fft)

    # 得到中心像素
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)

    # 构建掩模，256位，两个通道
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
            # 计算(i, j)到中心点的距离
            d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
            if radius - w / 2 < d < radius + w / 2:
                mask[i, j, 0] = mask[i, j, 1] = 0
            else:
                mask[i, j, 0] = mask[i, j, 1] = 1

    # 给傅里叶变换结果乘掩模
    fft_filtering = dshift * np.float32(mask)
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv.idft(ishift)
    image_filtering = cv.magnitude(image_filtering[:, :, 0],
                                   image_filtering[:, :, 1])
    # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
    cv.normalize(image_filtering, image_filtering, 0, 1, cv.NORM_MINMAX)
    return image_filtering


if __name__ == "__main__":
    image = cv.imread("pictures/lena.jpg", 0)
    image_noise = salt_pepper(image, 0.05, 0.05)
    image_low_pass_filtering5 = low_pass_filtering(image, 50)
    image_low_pass_filtering1 = low_pass_filtering(image, 10)
    image_high_pass_filtering5 = high_pass_filtering(image, 50, 1)
    image_high_pass_filtering1 = high_pass_filtering(image, 10, 1)
    image_bandpass_filtering5 = bandpass_filter(image, 30, 56, 1)
    image_bandstop_filtering5 = bandstop_filter(image, 30, 56, 1)
    plt.subplot(1, 4, 1), plt.imshow(
        image, 'gray'), plt.title("Origin"), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 4, 2), plt.imshow(
        image_noise, 'gray'), plt.title("Salt"), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 4, 3), plt.imshow(
    #     image_low_pass_filtering5,
    #     'gray'), plt.title("low-r-50"), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 4, 4), plt.imshow(
    #     image_low_pass_filtering1,
    #     'gray'), plt.title("low-r-10"), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 4, 3), plt.imshow(
    #     image_high_pass_filtering5,
    #     'gray'), plt.title("high-r-50"), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 4, 4), plt.imshow(
    #     image_high_pass_filtering1,
    #     'gray'), plt.title("high-r-10"), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 4, 3), plt.imshow(image_bandpass_filtering5,
    #                                  'gray'), plt.title("pass"), plt.xticks(
    #                                      []), plt.yticks([])
    # plt.subplot(1, 4, 4), plt.imshow(image_bandstop_filtering5,
    #                                  'gray'), plt.title("stop"), plt.xticks(
    #                                      []), plt.yticks([])
    plt.show()