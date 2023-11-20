import cv2
import numpy as np

class ImageProcess:
    # 用于产生图片的mask遮罩,在cv2.inRange中使用, 效果是将证件底的底色变白色
    colorLowerB = {
        'red': np.array([156, 43, 46]),
        'orange': np.array([11, 43, 46]),
        'yellow': np.array([26, 43, 46]),
        'green': np.array([35, 43, 46]),
        'cyan': np.array([78, 43, 46]),
        'blue': np.array([100, 43, 46]),
        'purple': np.array([125, 43, 46]),
        'black': np.array([0, 0, 0]),
        'gray': np.array([0, 0, 46]),
        'white': np.array([0, 0, 221]),
    }

    colorUpperB = {
        'red': np.array([180, 255, 255]),
        'orange': np.array([25, 255, 255]),
        'yellow': np.array([34, 255, 255]),
        'green': np.array([77, 255, 255]),
        'cyan': np.array([99, 255, 255]),
        'blue': np.array([124, 255, 255]),
        'purple': np.array([155, 255, 255]),
        'black': np.array([180, 255, 46]),
        'gray': np.array([180, 43, 220]),
        'white': np.array([180, 30, 255]),
    }
    # BGR通道
    destColor = {
        'red': (0, 0, 255),
        'orange': (0, 165, 255),
        'yellow': (0, 255, 255),
        'green': (0, 128, 0),
        'blue': (255, 0, 0),
        'cyan': (255, 255, 0),
        'purple': (128, 0, 128),
        'black': (0, 0, 0),
        'gray': (128, 128, 128),
        'white': (255, 255, 255),
    }

    def changeBackground(self, img, originColor='white', changeColor='blue'):
        originImg = self.copyImg(img)
        rows, cols, channels = img.shape

        # 图片转换为灰度图
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 图片的二值化处理
        lowerB = ImageProcess.colorLowerB.get(originColor)
        upperB = ImageProcess.colorUpperB.get(originColor)
        mask = cv2.inRange(hsv, lowerB, upperB)

        # 针对图片进行腐蚀膨胀
        kernel = np.ones((3, 3), np.uint8)
        # kernel = None
        erode = cv2.erode(mask, kernel, iterations=1)
        dilate = cv2.dilate(erode, kernel, iterations=1)
        # 遍历每个像素点，进行颜色的替换
        print(type(ImageProcess.destColor))
        tmpColor = ImageProcess.destColor.get(changeColor)
        print(tmpColor)
        for i in range(rows):
            for j in range(cols):
                if dilate[i, j] == 255:  # 像素点为255表示的是白色，我们就是要将白色处的像素点，替换为红色
                    img[i, j] = tmpColor  # 此处替换颜色，为BGR通道，不是RGB通道
        return img

    def copyImg(self, img):
        '''
        复制图像
        :param img:
        :return:
        '''
        return cv2.cvtColor(img, cv2.NORMAL_CLONE)


if __name__ == '__main__':
    imageProcess = ImageProcess()
    img1 = cv2.imread('zhl.jpg')
    # 缩放图片
    img1 = cv2.resize(img1, None, fx=0.2, fy=0.2)
    originImg1 = imageProcess.copyImg(img1)
    changeImg = imageProcess.changeBackground(img1, 'blue', 'yellow')
    cv2.imshow("yuantu", originImg1)
    cv2.imshow("jieguo", changeImg)
    # 窗口等待的命令，0表示无限等待
    cv2.waitKey(0)
