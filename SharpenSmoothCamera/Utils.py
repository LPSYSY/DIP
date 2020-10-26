import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


def CV2PIL(img):
    '''
    CV图像转换成PIL图像
    '''
    img = cv.resize(img, (300, 300), cv.INTER_CUBIC)
    img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    return img