'''
界面类
'''

import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import cv2 as cv
import ConstValue as con


class MainWindow(tk.Tk):
    def __init__(self):
        # 父类属性
        super().__init__()

        # 窗口初是属性
        self.geometry('800x600')
        self.title('Object Recognition')
        self.resizable(False, False)
        self.geometry("%dx%d+%d+%d" % (con.ww, con.wh, con.x, con.y))
        self.picNameLabel = tk.Label(self, text="The name of your picture: ")
        self.pictureName = tk.Entry(self)

    def initPictures(self, photoLeft, photoRight):
        # 初始化界面图像
        self.LabLeft = tk.Label(self, compound='center', image=photoLeft)
        self.LabLeft.place(x=50, y=50)
        self.LabRight = tk.Label(self, compound='center', image=photoRight)
        self.LabRight.place(x=440, y=50)
        self.picNameLabel.place(x=200, y=400)
        self.pictureName.place(x=360, y=400)

    # 显示窗口有函数
    def show(self):
        self.mainloop()
