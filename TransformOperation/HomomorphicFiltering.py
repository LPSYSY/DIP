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


class HomomorphicFilter:
    # 初始化
    def __init__(self, a=0.5, b=1.5):
        self.a = float(a)
        self.b = float(b)

    # 构造高斯滤波器
    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]),
                           range(I_shape[1]),
                           sparse=False,
                           indexing='ij')
        Duv = (((U - P)**2 + (V - Q)**2)).astype(float)
        H = np.exp((-Duv / (2 * (filter_params[0])**2)))
        return (1 - H)

    # 使用滤波器返回处理后的图像
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b * H) * I
        return I_filtered

    def filter(self, I, filter_params, filter='gaussian', H=None):
        # 判断是否是灰度图
        if len(I.shape) is not 2:
            raise Exception('Improper image')
        # 取对数
        I_log = np.log1p(np.array(I, dtype="float"))
        # FFT
        I_fft = np.fft.fft2(I_log)
        # 滤波器
        H = self.__gaussian_filter(I_shape=I_fft.shape,
                                   filter_params=filter_params)
        # 使用滤波器
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        # 反傅里叶
        I_filt = np.fft.ifft2(I_fft_filt)
        # 指数 还原
        I = np.exp(np.real(I_filt)) - 1
        # 取整
        return np.uint8(I)


img = cv.imread('pictures/lena.jpg', 0)
homo_filter = HomomorphicFilter(a=0.75, b=1.25)
img_filtered = homo_filter.filter(I=img, filter_params=[30, 2])
cv_show('res', showTwoPics(img, img_filtered))