import cv2
import numpy as np


def img_show(title, image):
    """使用opencv显示图片

    Args:
        title (string): 窗口名
        image (cv2.image): 原始图片
    """
    h, w = image.shape[:2]
    max_h = 900
    if h > max_h:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, max_h * w // h, max_h)
    cv2.imshow(title, image)


def remove_shadow_color(img):
    """去除彩色图片背景中的阴影

    Args:
        img (cv2.image): 原始彩色图片

    Returns:
        cv2.image: 去除阴影的彩色图片
    """
    rgb_planes = cv2.split(img)  # 分离 RGB 通道

    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))  # 对通道进行膨胀操作
        bg_img = cv2.medianBlur(dilated_img, 21)  # 对膨胀后的图像进行中值滤波
        diff_img = 255 - cv2.absdiff(plane, bg_img)  # 获取阴影差异图像
        norm_img = cv2.normalize(
            diff_img,
            None,
            alpha=0,  # 输出数组取值范围最小值
            beta=255,  # 输出数组取值范围最大值
            norm_type=cv2.NORM_MINMAX,  # 归一化类型
        )
        result_norm_planes.append(norm_img)  # 将归一化后的差异图像添加到列表

    shadow_free_image = cv2.merge(result_norm_planes)  # 合并三个通道
    return shadow_free_image  # 返回去除阴影的图像


def remove_shadow_gray(image, blur_size, mode=0):
    """去除灰度图片背景中的阴影，使用滤波方法去除图片中的阴影部分

    Args:
        image (cv2.image): 原始灰度图片
        blur_size (_type_): 滤波块的尺寸
        mode (int, optional): 去除阴影模式，0 用于浅色背景，深色阴影；1 用于深色背景，浅色阴影. Defaults to 0.

    Returns:
        _type_: 去除阴影后的图片
    """
    if mode == 0:
        # mode 0 用于浅色背景，深色阴影
        bg_image = _max_filtering(image=image, size=blur_size)
        bg_image = _min_filtering(image=bg_image, size=blur_size)
    elif mode == 1:
        # mode 1 用于深色背景，浅色阴影
        bg_image = _min_filtering(image=image, size=blur_size)
        bg_image = _max_filtering(image=bg_image, size=blur_size)
    normalised_img = cv2.normalize(image - bg_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(normalised_img)


def smooth_binarization(image, blur_size=5):
    """平滑OTSU二值化图片，保证文字平滑的前提下生成二值化灰度图片

    Args:
        image (cv2.image): 原始灰度图片
        blur_size (int, optional): 高斯平滑的滑块尺寸. Defaults to 5.

    Returns:
        cv2.image: 返回处理后的灰度图片
    """
    h, w = image.shape[:2]
    # 放大图片
    image = cv2.resize(image, (2 * w, 2 * h), interpolation=cv2.INTER_CUBIC)
    # 锐化
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)
    # 平滑过滤
    image = cv2.GaussianBlur(image, (blur_size + 2, blur_size + 2), 0)
    # 二值化
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 平滑过滤
    image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    # 还原图片
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    return image


def _max_filtering(image, size):
    # 添加边框
    half_size = size // 2
    # 将图像转换为float32类型
    image = image.astype('float32')
    # 将-1替换成NaN
    image[image == -1] = np.nan
    # 进行边框填充
    border_image = cv2.copyMakeBorder(image,
                                      half_size,
                                      half_size,
                                      half_size,
                                      half_size,
                                      cv2.BORDER_CONSTANT,
                                      value=0)
    # 最大滤波操作
    kernel = np.ones((size, size), np.uint8)
    dilated_image = cv2.dilate(border_image, kernel, iterations=1)
    # 将NaN替换成-1
    dilated_image[np.isnan(dilated_image)] = -1
    # 将结果转换为int类型
    dilated_image = dilated_image.astype('int')
    # 去除填充像素，返回输出结果
    result = dilated_image[half_size:-half_size, half_size:-half_size]
    return result


def _min_filtering(image, size):
    # 计算结构元素半径
    half_size = size // 2
    # 将图像转换为float32类型
    image = image.astype('float32')
    # 将300替换成np.nan(占位符)
    image[image == 300] = np.nan
    # 边缘填充
    border_image = cv2.copyMakeBorder(image,
                                      half_size,
                                      half_size,
                                      half_size,
                                      half_size,
                                      cv2.BORDER_CONSTANT,
                                      value=255)
    # 创建结构元素
    kernel = np.ones((size, size), np.uint8)
    # 单次腐蚀 相当于最小滤波
    eroded_image = cv2.erode(border_image, kernel, iterations=1)
    # 将占位符替换成300
    eroded_image[np.isnan(eroded_image)] = 300
    # 将结果转换为int类型
    eroded_image = eroded_image.astype('int')
    # 去除填充部分，得到输出结果
    result = eroded_image[half_size:-half_size, half_size:-half_size]
    return result