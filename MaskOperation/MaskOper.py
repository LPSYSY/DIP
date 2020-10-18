import cv2 as cv
import numpy as np


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


def ImageSmooth(img, fileterType, kernel=3):
    imgOrigin = img
    # 均值滤波
    if fileterType == 'average':
        imgResult = cv.blur(imgOrigin, (kernel, kernel))
        cv_show('res', showTwoPics(imgOrigin, imgResult))
    # 中值滤波
    elif fileterType == 'median':
        imgResult = cv.medianBlur(imgOrigin, kernel)
        cv_show('res', showTwoPics(imgOrigin, imgResult))


def ImageSharpen(img, filerType, kernel=3):
    imgOrigin = img
    if filerType == 'Sobel':
        # Sobel算子图像锐化
        imgResultX = cv.Sobel(imgOrigin, cv.CV_64F, 1, 0, ksize=kernel)
        imgResultY = cv.Sobel(imgOrigin, cv.CV_64F, 0, 1, ksize=kernel)
        imgResultX = cv.convertScaleAbs(imgResultX)
        imgResultY = cv.convertScaleAbs(imgResultY)
        imgResult = cv.addWeighted(imgResultX, 0.5, imgResultY, 0.5, 0)
        cv_show('res', showTwoPics(imgOrigin, imgResult))
    elif filerType == 'Laplace':
        imgResult = cv.Laplacian(img, cv.CV_64F)
        imgResult = cv.convertScaleAbs(imgResult)
        cv_show('res', showTwoPics(imgOrigin, imgResult))


if __name__ == "__main__":
    imgOrigin = cv.imread('pictures/lena.jpg', 0)
    # ImageSmooth(imgOrigin, 'average', 5)
    # ImageSmooth(imgOrigin, 'median', 5)
    # ImageSharpen(imgOrigin, 'Sobel', 3)
    ImageSharpen(imgOrigin, 'Laplace', 3)