#Пропишем алгоритмы приведения изображения к нормальной форме 
import cv2 
import matplotlib.pyplot as plt
import imageio
import numpy as np
from sympy import ITE 
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL as pil
from PIL import Image, ImageFilter
def edges(image_path):
    image= Image.open(image_path)
    edges=image.filter(ImageFilter.FIND_EDGES)
    #plt.imshow(edges)
    #plt.show()
    return edges

def non_shum(image_path):
    #matrix_analyze,matrix=np.asarray(image, dtype='uint8'),[np.asarray(image, dtype='uint8')]
    #column=len(matrix[0])
    #row=len(matrix[0][0])
    #matrix=np.asarray(image, dtype='uint8')

    img = plt.imread(f'{image_path}')
    plt.imshow(img,interpolation='none')
    plt.show()
    img_bw = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype('uint8')


    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    masking = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel_close)
    masking = cv2.morphologyEx(masking, cv2.MORPH_OPEN, kernel_open)
   
    masking = cv2.dilate(masking, kernel_open, iterations=1)

    masking = cv2.morphologyEx(masking, cv2.MORPH_GRADIENT, kernel_open)
    
    masking = cv2.dilate(masking, kernel_open, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    masking = cv2.morphologyEx(masking, cv2.MORPH_CLOSE, kernel_close)
    masking = cv2.morphologyEx(masking, cv2.MORPH_OPEN, kernel_open)
    #________
    # Метод разделяй и властвуй раздели на минимально возможные блоки и начни обработку как сектора если сектор более пустой чем средне квадратичное удаляй все
    
    #cort=img_bw.shape
    #vs,sh,dht=cort(0),cort(1),cort(2)

        
    

    #________

    masking = cv2.dilate(masking, kernel_open, iterations=2)
    
    plt.imshow(masking,interpolation='none')
    plt.show()

    return masking


def normalization(image_path):
    #normal_image=non_shum(edges(image_path))
    
    return non_shum(image_path)

def morphological_change_image(image):
    kernel_open =cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    kernel_close =cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    masking=cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_close)
    masking = cv2.morphologyEx(masking, cv2.MORPH_OPEN, kernel_open)
   
    masking=cv2.dilate(masking, kernel_open, iterations=1)

    masking=cv2.morphologyEx(masking, cv2.MORPH_GRADIENT, kernel_open)

    masking=cv2.dilate(masking, kernel_open, iterations=2)

    kernel_open =cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    kernel_close =cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    masking = cv2.morphologyEx(masking, cv2.MORPH_CLOSE, kernel_close)
    masking = cv2.morphologyEx(masking, cv2.MORPH_OPEN, kernel_open)

    masking=cv2.dilate(masking, kernel_open, iterations=2)
    return masking

def image_view(image_path: str):
    image = plt.imread(f'{image_path}')
    plt.imshow(image)
    plt.show()

    return True

if __name__ == '__main__':
    normalization(f'D:\Python\Dataset-original\\2.bmp')
    #image_view('D:\Python\Dataset-original\\2.bmp')

