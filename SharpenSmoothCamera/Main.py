import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import cv2 as cv
import ConstValue as con
import Interface as interface
import TransPictures as tp
import Utils as utils

mainwindow = interface.MainWindow()

photoLeft = ImageTk.PhotoImage(con.imgLeft)
photoRight = ImageTk.PhotoImage(con.imgRight)
img = cv.imread('pictures/01.jpg')


def GetPicName():
    '''
    获取图片名
    '''
    name = "pictures/" + mainwindow.pictureName.get()
    global img
    img = cv.imread(name)


# 输入图片名的按键
buttonInput = tk.Button(mainwindow,
                        text='Input',
                        bg='blue',
                        command=GetPicName)
buttonInput.place(x=530, y=400, width=100, height=25)


def PictureAverFilter():

    global photoNew
    photoNew = ImageTk.PhotoImage(utils.CV2PIL(tp.averFilter(img)))
    LabReplace = tk.Label(mainwindow, compound='center', image=photoNew)
    LabReplace.place(x=440, y=50)


def PictureMedFilter():

    global photoNew
    photoNew = ImageTk.PhotoImage(utils.CV2PIL(tp.medianFilter(img)))
    LabReplace = tk.Label(mainwindow, compound='center', image=photoNew)
    LabReplace.place(x=440, y=50)


def PictureGauFilter():

    global photoNew
    photoNew = ImageTk.PhotoImage(utils.CV2PIL(tp.GuassFilter(img)))
    LabReplace = tk.Label(mainwindow, compound='center', image=photoNew)
    LabReplace.place(x=440, y=50)


def PictureTwoFilter():

    global photoNew
    photoNew = ImageTk.PhotoImage(utils.CV2PIL(tp.TwoEdgesFilter(img)))
    LabReplace = tk.Label(mainwindow, compound='center', image=photoNew)
    LabReplace.place(x=440, y=50)


def PictureSobelSharpen():

    global photoNew
    photoNew = ImageTk.PhotoImage(utils.CV2PIL(tp.SobelSharpen(img)))
    LabReplace = tk.Label(mainwindow, compound='center', image=photoNew)
    LabReplace.place(x=440, y=50)


def PictureLapSharpen():

    global photoNew
    photoNew = ImageTk.PhotoImage(utils.CV2PIL(tp.LaplaceSharpen(img)))
    LabReplace = tk.Label(mainwindow, compound='center', image=photoNew)
    LabReplace.place(x=440, y=50)


def PictureScharrSharpen():

    global photoNew
    photoNew = ImageTk.PhotoImage(utils.CV2PIL(tp.ScharrSharpen(img)))
    LabReplace = tk.Label(mainwindow, compound='center', image=photoNew)
    LabReplace.place(x=440, y=50)


def PictureCannySharpen():

    global photoNew
    photoNew = ImageTk.PhotoImage(utils.CV2PIL(tp.CannySharpen(img)))
    LabReplace = tk.Label(mainwindow, compound='center', image=photoNew)
    LabReplace.place(x=440, y=50)


# 平滑按钮设置

# 均值滤波
buttonAverFilter = tk.Button(mainwindow,
                             text='AverageFilter',
                             bg='red',
                             command=PictureAverFilter)

buttonAverFilter.place(x=40, y=450, width=120, height=50)

# 中值滤波
buttonMedFilter = tk.Button(mainwindow,
                            text='MedianFilter',
                            bg='orange',
                            command=PictureMedFilter)

buttonMedFilter.place(x=240, y=450, width=120, height=50)

# 高斯滤波
buttonGauFilter = tk.Button(mainwindow,
                            text='GuassFilter',
                            bg='yellow',
                            command=PictureGauFilter)

buttonGauFilter.place(x=440, y=450, width=120, height=50)

# 双边滤波
buttonTwoFilter = tk.Button(mainwindow,
                            text='bilateralFilter',
                            bg='green',
                            command=PictureTwoFilter)

buttonTwoFilter.place(x=640, y=450, width=120, height=50)

# Sobel
buttonSobelSharpen = tk.Button(mainwindow,
                               text='SobelSharpen',
                               bg='red',
                               command=PictureSobelSharpen)

buttonSobelSharpen.place(x=40, y=510, width=120, height=50)

# Laplace
buttonLaplaceSharpen = tk.Button(mainwindow,
                                 text='LaplaceSharpen',
                                 bg='orange',
                                 command=PictureLapSharpen)

buttonLaplaceSharpen.place(x=240, y=510, width=120, height=50)

# Scharr
buttonScharrSharpen = tk.Button(mainwindow,
                                text='ScharrSharpen',
                                bg='yellow',
                                command=PictureScharrSharpen)

buttonScharrSharpen.place(x=440, y=510, width=120, height=50)

# Canny
buttonCannySharpen = tk.Button(mainwindow,
                               text='CannySharpen',
                               bg='green',
                               command=PictureCannySharpen)

buttonCannySharpen.place(x=640, y=510, width=120, height=50)

mainwindow.initPictures(photoLeft, photoRight)
mainwindow.show()
