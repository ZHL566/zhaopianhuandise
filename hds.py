import cv2
import numpy as np
import concurrent.futures
import streamlit as st

st.title('基于OpenCV的图像分割系统')


def process_image(image, new_background_color):
    # 如果原图尺寸太大，可以进行图像压缩
    scale_percent = 50  # 压缩比例（50%）

    # 计算压缩后的图像尺寸
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # 调整图像尺寸
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # 进行图像处理
    result = grabcut_segmentation(resized_image, new_background_color)

    # 返回处理结果
    return result


def grabcut_segmentation(image, new_background_color):
    # 创建掩膜
    mask = np.zeros(image.shape[:2], np.uint8)

    # 定义GrabCut算法所需的前景和背景模型
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 定义矩形区域，包含前景对象（根据实际需要调整矩形位置和大小）
    height, width = image.shape[:2]
    rect = (10, 10, width - 10, height - 10)

    # 执行GrabCut算法, 通过调整num_iterations参数来控制迭代次数，以平衡速度和准确性
    num_iterations = 5
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, num_iterations, cv2.GC_INIT_WITH_RECT)

    # 创建前景和背景掩膜
    mask_foreground = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask_background = 1 - mask_foreground

    # 创建纯色背景图像
    background = np.zeros_like(image, np.uint8)
    background[:] = new_background_color

    # 将前景放在白色背景上
    foreground = cv2.bitwise_and(image, image, mask=mask_foreground)
    result = cv2.add(foreground, cv2.bitwise_and(background, background, mask=mask_background))

    return result


# 定义常用背景颜色，注意OpenCV 颜色通道顺序为 B G R
blue_BGR = (240, 167, 2)  # 蓝色
white_BGR = (255, 255, 255)  # 白色
red_BGR = (27, 0, 217)  # 红色
blue_white_BGR = (196, 146, 52)  # 蓝白渐变色
light_gray_BGR = (210, 210, 210)  # 浅灰色


uploaded_file = st.file_uploader("选择你需要换底色的图片", type=("jpg", "png"))
if uploaded_file is not None:
    #将传入的文件转为Opencv格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_path = cv2.imdecode(file_bytes, 1)
    #展示图片
    st.image(image_path, channels="BGR")
    cv2.imwrite('test.jpg', image_path)
# 读取原图像
image = cv2.imread('test.jpg')

new_background_color = white_BGR  # 白色背景色
# 使用多线程处理单张照片
with concurrent.futures.ThreadPoolExecutor() as executor:
    result = executor.submit(process_image, image, new_background_color)
if st.button('白底'):
    # 获取处理结果
    result = result.result()
    st.write("白色背景：")
    st.image(result, channels="BGR")

new_background_color = blue_BGR  # 蓝色背景色
# 使用多线程处理单张照片
with concurrent.futures.ThreadPoolExecutor() as executor:
    result = executor.submit(process_image, image, new_background_color)
if st.button('蓝底'):
    # 获取处理结果
    result = result.result()
    st.write("蓝色背景：")
    st.image(result, channels="BGR")


new_background_color = red_BGR  # 红色背景色
# 使用多线程处理单张照片
with concurrent.futures.ThreadPoolExecutor() as executor:
    result = executor.submit(process_image, image, new_background_color)
if st.button('红底'):
    # 获取处理结果
    result = result.result()
    st.write("红色背景：")
    st.image(result, channels="BGR")


new_background_color = blue_white_BGR  # 蓝白背景色
# 使用多线程处理单张照片
with concurrent.futures.ThreadPoolExecutor() as executor:
    result = executor.submit(process_image, image, new_background_color)
if st.button('蓝白底'):
    # 获取处理结果
    result = result.result()
    st.write("蓝白色背景：")
    st.image(result, channels="BGR")


new_background_color = light_gray_BGR  # 浅灰背景色
# 使用多线程处理单张照片
with concurrent.futures.ThreadPoolExecutor() as executor:
    result = executor.submit(process_image, image, new_background_color)
if st.button('灰白底'):
    # 获取处理结果
    result = result.result()
    st.write("灰白色背景：")
    st.image(result, channels="BGR")
    # cv2.waitKey(0)
