import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def MyHe(img):
    # 统计直方图每个灰度级出现的次数
    calDic = np.zeros(256, dtype=int)

    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            calDic[img[x][y]] += 1

    # 进行归一化·
    p = [i for i in range(256)]
    for i in range(0, 256):
        p[i] = calDic[i] / float(img.size)

    # 累积
    c = [i for i in range(256)]
    c[0] = p[0]
    for i in range(1, 256):
        c[i] = c[i - 1] + p[i]

    # 应用公式找到映射值
    PixelMap = [0 for i in range(256)]
    for i in range(256):
        PixelMap[i] = int((256 - 1) * c[i] + 0.5)

    # print(PixelMap)

    # 创建一副新的空图像
    newImg = np.zeros(img.shape, dtype=np.uint8)
    # print(newImg)

    # 根据映射创建新的图像
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            newImg[i][j] = PixelMap[img[i][j]]

    return newImg


# Test:

# 读入图像
img = cv.imread('pictures/lena.jpg', 0)

newImg = MyHe(img)

before = cv.calcHist([img], [0], None, [256], [0, 256])
after = cv.calcHist([newImg], [0], None, [256], [0, 256])
plt.plot(before)
plt.plot(after)
plt.xlim([0, 256])
plt.show()

cv.imshow('img', img)
cv.imshow('newImg', newImg)
cv.waitKey(0)
cv.destroyAllWindows()