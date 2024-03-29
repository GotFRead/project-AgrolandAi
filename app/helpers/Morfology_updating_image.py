# Пропишем алгоритмы приведения изображения к нормальной форме
import math
import cv2
import cv2 as morfological_transformation
import matplotlib.pyplot as plt
import imageio
import numpy as np
# from sympy import ITE
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL as pil
from PIL import Image, ImageFilter
import base64
import os


def convert_to_channel_one_channel(path_to_image: str):
    return cv2.imread(path_to_image, 0)


def edges(image_path):
    image = Image.open(image_path)
    edges = image.filter(ImageFilter.FIND_EDGES)
    # plt.imshow(edges)
    # plt.show()
    return edges


def change_format_image():
    pass


def dilate_image(image, iteratrion, size_kernal):
    kernal = morfological_transformation.getStructuringElement(
        morfological_transformation.MORPH_RECT, (size_kernal, size_kernal))
    image = morfological_transformation.dilate(
        image, kernal, iterations=iteratrion)
    return image


def non_shum(image_path):
    img = plt.imread(f'{image_path}')
    plt.imshow(img, interpolation='none')
    plt.show()
    img_bw = (morfological_transformation.cvtColor(
        img, morfological_transformation.COLOR_BGR2GRAY)).astype('uint8')

    kernel_open = morfological_transformation.getStructuringElement(
        morfological_transformation.MORPH_RECT, (2, 2))
    kernel_close = morfological_transformation.getStructuringElement(
        morfological_transformation.MORPH_RECT, (3, 3))
    masking = morfological_transformation.morphologyEx(
        img_bw, morfological_transformation.MORPH_CLOSE, kernel_close)
    masking = morfological_transformation.morphologyEx(
        masking, morfological_transformation.MORPH_OPEN, kernel_open)

    masking = morfological_transformation.dilate(
        masking, kernel_open, iterations=1)

    masking = morfological_transformation.morphologyEx(
        masking, morfological_transformation.MORPH_GRADIENT, kernel_open)

    masking = morfological_transformation.dilate(
        masking, kernel_open, iterations=2)

    kernel_open = morfological_transformation.getStructuringElement(
        morfological_transformation.MORPH_RECT, (1, 1))
    kernel_close = morfological_transformation.getStructuringElement(
        morfological_transformation.MORPH_RECT, (2, 2))
    masking = morfological_transformation.morphologyEx(
        masking, morfological_transformation.MORPH_CLOSE, kernel_close)
    masking = morfological_transformation.morphologyEx(
        masking, morfological_transformation.MORPH_OPEN, kernel_open)
    # ________
    # TODO Метод разделяй и властвуй раздели на минимально возможные блоки и начни обработку как сектора если сектор более пустой чем средне квадратичное удаляй все

    # cort=img_bw.shape
    # vs,sh,dht=cort(0),cort(1),cort(2)

    # ________

    masking = morfological_transformation.dilate(
        masking, kernel_open, iterations=2)

    plt.imshow(masking, interpolation='none')
    plt.show()

    return masking


def normalization(image_path):
    # normal_image=non_shum(edges(image_path))

    return non_shum(image_path)


def morphological_change_image(image):
    kernel_open = morfological_transformation.getStructuringElement(
        morfological_transformation.MORPH_RECT, (2, 2))
    kernel_close = morfological_transformation.getStructuringElement(
        morfological_transformation.MORPH_RECT, (3, 3))
    masking = morfological_transformation.morphologyEx(
        image, morfological_transformation.MORPH_CLOSE, kernel_close)
    masking = morfological_transformation.morphologyEx(
        masking, morfological_transformation.MORPH_OPEN, kernel_open)

    masking = morfological_transformation.dilate(
        masking, kernel_open, iterations=1)

    kernel_open = morfological_transformation.getStructuringElement(
        morfological_transformation.MORPH_RECT, (1, 1))
    kernel_close = morfological_transformation.getStructuringElement(
        morfological_transformation.MORPH_RECT, (2, 2))
    masking = morfological_transformation.morphologyEx(
        masking, morfological_transformation.MORPH_CLOSE, kernel_close)
    masking = morfological_transformation.morphologyEx(
        masking, morfological_transformation.MORPH_OPEN, kernel_open)
    masking = morfological_transformation.dilate(
        masking, kernel_open, iterations=1)

    return masking


def morphological_dilate_image(image):
    kernel_open = morfological_transformation.getStructuringElement(
        morfological_transformation.MORPH_RECT, (5, 5))
    kernel_close = morfological_transformation.getStructuringElement(
        morfological_transformation.MORPH_RECT, (7, 7))

    masking = morfological_transformation.dilate(
        image, kernel_open, iterations=10)

    masking = morfological_transformation.erode(
        masking, kernel_close, iterations=3)

    return masking


def image_from_bytearray(byte_string: bytearray):
    array_numpy = np.array(byte_string.decode('utf-8'))


def image_view(image_path: str):
    image = plt.imread(f'{image_path}')
    plt.imshow(image)
    plt.show()

    return True


def cleaner_to_value(image, params):
    summary_value_px, count_px = int(), int()
    for x in range(len(image)):
        for y in range(len(image[x])):
            for px in image[x][y]:
                summary_value_px += px
                count_px += 1

    denose = float(summary_value_px / count_px)

    for x in range(len(image)):
        for y in range(len(image[x])):
            image[x][y] -= [denose, denose, denose]

    for x in range(len(image)):
        for y in range(len(image[x])):
            temp = list()
            for u in image[x][y]:
                temp.append(math.ceil(u))

            image[x][y] = temp

    # return image

    # plt.imshow(image)
    # plt.show()
    # image = (morfological_transformation.cvtColor(image, morfological_transformation.COLOR_BGR2GRAY)).astype('uint8')
    return image


def save_morfological_transformation_image(name, image):
    name = name.split('.')[0]
    name = name.split('-')[3]
    morfological_transformation.imwrite(f'{name}.png', img=image*255)


def wait_byte_form_image(bytestring):
    if b'IHDR' in bytestring and b'IEND\xaeB`\x82' in bytestring:
        with open(r"D:\Python\Project_PVS-CNN\склон-1 - демонстрационная выборка\cgfd.png", 'wb') as img:
            image = img.write(bytestring)
        return True
    return False


def normalization_by_dirs(path, level_denoise=0.5):
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            os.chdir(r"D:\Python\datasets_from_app\normalized_image\равнины")
            image = plt.imread(root + f'\{file}')
            image = cleaner_to_value(image, level_denoise)
            image = morphological_change_image(image)
            name_image = file.replace('.', '_').split('_')[-2]
            print(root)
            morfological_transformation.imwrite(
                f'Normalized_plain{name_image}.png', image * 255)

def convert_to_one_channel_by_dirs(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            os.chdir(r'D:\Python\Project_PVS-CNN\one_channel_images\test_validate\slope')
            one_channel_image = convert_to_channel_one_channel(root + f'\{file}')
            name_image = file.split('.')[0]
            morfological_transformation.imwrite(
                f'{name_image}.png', one_channel_image)

def resize_image(path, scale: tuple): 
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            os.chdir(r'D:\Python\DataFrame\resize_one_channel\train\plain')
            resize_one_channel_image = plt.imread(root + f'\{file}')
            resize_one_channel_image = cv2.resize(resize_one_channel_image, scale)
            morfological_transformation.imwrite(
                f"r_{file.split('.')[0]}.png", resize_one_channel_image)
                # name_image = file.split('.')[0]
                # morfological_transformation.imwrite(
                #     f'{name_image}.png', resize_one_channel_image)

if __name__ == '__main__':
    pass
