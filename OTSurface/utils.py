import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# The shelf's layers
def LayersNum(imgName):
    # print('LayersNum: ' + str(imgName))
        imgOrigin = cv.imread(imgName)
        # Sobel算子图像锐化
        imgResultX = cv.Sobel(imgOrigin, cv.CV_64F, 1, 0, ksize=3)
        imgResultY = cv.Sobel(imgOrigin, cv.CV_64F, 0, 1, ksize=3)
        imgResultX = cv.convertScaleAbs(imgResultX)
        imgResultY = cv.convertScaleAbs(imgResultY)
        imgResult = cv.addWeighted(imgResultX, 0.5, imgResultY, 0.5, 0)
        return imgResult

# Quantity of Goods
def GoodsQuantity(imgName):
    print('GoodsQuantity: ' + str(imgName))
    

# Color of these Goods
def GoodsColor(imgName):
    print('GoodsColor: ' + str(imgName))


if __name__ == "__main__":
    pass