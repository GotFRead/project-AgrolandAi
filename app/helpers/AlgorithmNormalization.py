#Пропишем алгоритмы приведения изображения к нормальной форме 
import cv2 
import matplotlib.pyplot as plt
import imageio
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL as pil
from PIL import Image, ImageFilter

#
# Development suspended
#


'''
def _FilterSobel_():
    sobel_y = np.array([[ -1, -2, -1], 
    [ 0, 0, 0], 
    [ 1, 2, 1]])
    sobel_x = np.array([[ -1, 0, 1], 
    [ 0, 0, 0], 
    [ 1, 2, 1]])
    filtered_image = cv.filter2D(gray, -1, sobel_y)
    $.imshow(filtered_image, cmap='gray')
'''

def edges(image_path):
    image= Image.open(image_path)
    edges=image.filter(ImageFilter.FIND_EDGES)
    plt.imshow(edges)
    plt.show()
    return edges

def non_shum(image):
    matrix_analyze,matrix=np.asarray(image, dtype='uint8'),[np.asarray(image, dtype='uint8')]
    #plt.imshow(matrix_analyze,interpolation='none')
    #plt.show()
    column=len(matrix[0])
    row=len(matrix[0][0])
    matrix=np.asarray(image, dtype='uint8')
    #matrix3x3=[[1,1,1],[1,1,1],[1,1,1]]
    #matrix5x5=[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
    #matrix7x7=[[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1]]
    '''for x in range(1,column):
        if x==column-1:
            break
        for y in range(1,row):
            if y==row-1:
                break
            if matrix[x][y]==1 and (matrix[x][y-1]==1 # Х У это центр матрицы 
            or matrix[x-1][y-1]==1 
            or matrix[x-1][y]==1 or matrix[x-1][y+1]==1 or matrix[x][y+1]==1 
            or matrix[x+1][y+1]==1 or matrix[x+1][y]==1 or matrix[x+1][y-1]==1) :  
                matrix_analyze[x][y]=matrix[x][y]
            else:  matrix_analyze[x][y]=0'''
    """
    for x in range(1,column):
        if x==column-1:
            break
        for y in range(1,row):
            if y==row-4:
                break
            if matrix[x][y]==1 and (matrix[x][y-1]==1 # Х У это центр матрицы 
            or matrix[x-1][y-1]==1 
            or matrix[x-1][y]==1 
            or matrix[x-1][y+1]==1) and ( matrix[x][y+1]==1 
            or matrix[x+1][y+1]==1 
            or matrix[x+1][y]==1 
            or matrix[x+1][y-1]==1):
                if   matrix[x-2][y]==1 and (matrix[x-2][y-1]==1 # Х У это центр матрицы 
            or matrix[x-2][y-2]==1 
            or matrix[x-1][y-2]==1 
            or matrix[x][y-2]==1) and (matrix[x+1][y-2]==1 
            or matrix[x+1][y-2]==1 
            or matrix[x+2][y-2]==1 
            or matrix[x+2][y-1]==1) and (matrix[x+2][y]==1 
            or matrix[x+2][y+1]==1 
            or matrix[x+2][y+2]==1 
            or matrix[x+1][y+2]==1) and (matrix[x][y+2]==1 
            or matrix[x-1][y+2]==1 
            or matrix[x-2][y+2]==1 
            or matrix[x-2][y+1]==1):
                    matrix_analyze[x][y]=1
                else: matrix[x][y]=1 
                matrix[x][y-1]=1 # Х У это центр матрицы 
                matrix[x-1][y-1]=1 
                matrix[x-1][y]=1 
                matrix[x-1][y+1]=1 
                matrix[x][y+1]=1 
                matrix[x+1][y+1]=1 
                matrix[x+1][y]=1 
                matrix[x+1][y-1]=1
            else:  matrix_analyze[x][y]=0ф
    print(x)    """
    img = plt.imread('D:\Python\Dataset-original\\2.bmp')
    img_bw = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')


    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

    mask = np.dstack([mask, mask, mask]) / 255

    plt.imshow(mask,interpolation='bilinear')
    plt.show()
'''
    plt.imshow(matrix_analyze,interpolation='none')
    plt.show()
    image=Image.fromarray(matrix_analyze,'RGB')
    image.show()'''


def normalization(image_path):
    normal_image=non_shum(edges(image_path))


normalization(f'D:\Python\Dataset-original\\2.bmp')