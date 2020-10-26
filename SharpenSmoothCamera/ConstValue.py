'''
常量
'''

import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# window = tk.Tk()
# sw = window.winfo_screenwidth()
# sh = window.winfo_screenheight()
# print(sw, sh)
# 屏幕宽高 窗口宽高 计算中心点位
sw, sh, ww, wh = 1368, 912, 800, 600
x = (sw - ww) / 2
y = (sh - wh) / 2

# 初始图片
path = ['pictures/01.jpg', 'pictures/02.jpg', 'pictures/03.jpg']
imgLeft = cv.imread(path[0])
imgLeft = cv.resize(imgLeft, (300, 300), cv.INTER_CUBIC)
imgLeft = Image.fromarray(cv.cvtColor(imgLeft, cv.COLOR_BGR2RGB))
imgRight = cv.imread(path[1])
imgRight = cv.resize(imgRight, (300, 300), cv.INTER_CUBIC)
imgRight = Image.fromarray(cv.cvtColor(imgRight, cv.COLOR_BGR2RGB))
imgNew = cv.imread(path[2])
imgNew = cv.resize(imgNew, (300, 300), cv.INTER_CUBIC)
imgNew = Image.fromarray(cv.cvtColor(imgNew, cv.COLOR_BGR2RGB))