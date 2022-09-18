#Пропишем алгоритмы приведения изображения к нормальной форме 
from asyncio import threads
from cgi import test
from multiprocessing import Event
import cv2 
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import stringStart
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL as pil
from PIL import Image, ImageFilter
#import testAsyncCalculated as test
import threading as thr
#import testLBP-analyze as test

#MatrixOfThreads=[[],[]] 

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

#class ThreadingOfAnalyze(object):
#    
#    def __init__(self,eventsForWait, eventsForSet,numberOfThreading,length,height):
#        self.eventsForWait=eventsForWait
#        self.eventsForSet=eventsForSet
#        self.numberOfThreading=numberOfThreading
#        self.length=length
#        self.height=height
#
#    def InitiateThreadings(self,threadsAnalyze,length,height):
#        threadsAnalyze= thr.Thread(target= LBP, args=(length,height))
#
#def StartEvents():
#    startEvent=thr.Event()
#    startEvent.set()
#
#    return startEvent

#    def StartThreading(self,threadsAnalyze):
#        threadsAnalyze.start()
#
#    def JoinThreads(self,threadsAnalyze):
#        threadsAnalyze.join()
#    
#    def CreateThreading(self,eventsForWait, eventsForSet,numberOfThreading):
#        for i in range(int(numberOfThreading)):
#            eventsForWait.wait()
#            eventsForWait.clear()
#            print(i)
#            eventsForSet.set() #set event for neighbor thead 


#Создать матрицу потоков что бы к ним обращаться 

#Создать 
'''
def CreateMatrix(numberThreads,length,height):
    numberThreads=int(numberThreads)
    print(f'Количество ридеров {numberThreads}')
    #ElementOfMatrix=(f'Количество потоков {numberThreads}', thr.Thread(target=LBP, args=(length,height)))
    #MatrixOfThreads=np.asfortranarray(ElementOfMatrix, )
    #Matrix= np.ones((ElementOfMatrix , numberThreads) , dtype=thr.Thread)
    Matrix= np.ones(numberThreads , dtype=thr.Thread)
    #[[ElementOfMatrix,0],[]]
    startEvent=thr.Event()
    startEvent.set()
    for schet in range(1,numberThreads):
        print(f'Запуск ридера:{schet}')
        Threads=thr.Thread(target= LBP, args=(length,height,schet))
        #print(type(Threads))
        #Threads.InitiateThreadings(Threads,length,height)
        #Threads.StartThreading()
        #Threads.JoinThreads(Threads)
        Threads.start()
        #Threads.join()      
        Matrix[schet]=Threads
def CreateThreads(ArrayForAnalyze, CountThreads):
    for schetThreads in range(1, CountThreads):
        for schetY in range(0,len(ArrayForAnalyze)):
            for 
 '''

def LBP(length,height,schet=12):
    print('Начало анализа')
    length,height= int(length), int(height)
    print(length,height)
    OffsetLenght=10*(schet%int(length*height/height))
    
    OffsetHeight=int(schet/int(length*height/height))

    print(OffsetLenght, OffsetHeight)

    LBPtemplate= np.array([[0,0,0],[0,0,0],[0,0,0]])
    multiplication= np.zeros(shape=(int(length),int(height)))
    for schetY in range(int(height*schet), int(height*(schet+1))):
        for schetX in range(length*schet, length*(schet+1)):
            print("Итераций",schetY)
            ArrayForMulti=length*height[schetX-1:schetX+2,schetY-1:schetY+2]
            multiplication[schetX-1:schetX+2,schetY-1:schetY+2]= ArrayForMulti* LBPtemplate 
            #print(multiplication)

    print(multiplication)
    
    '''
    cv2.imwrite('2.jpg',multiplication*255)

    s=cv2.imread('2.jpg')

    plt.imshow(s)
    plt.show()'''



def Separator(length,height):
    if length<500 and height<500:
        return length,height
    if length>height:
        length-=height
    elif height>length: 
        height-=length
    elif(length%2!=1 and height%2!=1): 
        length,height= length/2,height/2
    elif(length==height and length>500 and height>500):
        length-=height*3/5
    else: 
        print('Атомарный блок равен',height*length)        
        return length,height
    print(length,height)
    return Separator(length,height)


def SpeedCheck(lengthAnalyzeArray, heightAnalyzeArray,length,height):
    NumberOfThreading=0
    atomicBlock= length*height
    analyzeSegment=lengthAnalyzeArray*heightAnalyzeArray 
    print('Стандартный атомарный блок равен',atomicBlock)
    if (analyzeSegment>5000**2):
        if (atomicBlock<(analyzeSegment/atomicBlock)/1000):
                NumberOfThreading=1000
                #return atomicBlock, analyzeSegment/atomicBlock, NumberOfThreading
        else:
            print(atomicBlock)
            atomicBlock*=(analyzeSegment/atomicBlock)/1000
    elif(analyzeSegment>2500**2):
        if (atomicBlock<(analyzeSegment/atomicBlock)/500):
                NumberOfThreading=500
                #return atomicBlock, analyzeSegment/atomicBlock, NumberOfThreading
        else:
            print(atomicBlock)
            atomicBlock*=(analyzeSegment/atomicBlock)/500
    elif(analyzeSegment>1000**2):
        if (atomicBlock<(analyzeSegment/atomicBlock)/250):
                NumberOfThreading=250
                #return atomicBlock, analyzeSegment/atomicBlock, NumberOfThreading
        else: 
            print(atomicBlock)
            atomicBlock*=(analyzeSegment/atomicBlock)/250
    elif(analyzeSegment>500**2):
        if (atomicBlock<(analyzeSegment/atomicBlock)/100):
                NumberOfThreading=100
                #return atomicBlock, analyzeSegment/atomicBlock, NumberOfThreading
        else:
            print(atomicBlock)
            atomicBlock*=(analyzeSegment/atomicBlock)/100
    elif(analyzeSegment>100**2):
        if (atomicBlock<(analyzeSegment/atomicBlock)/50):
                NumberOfThreading=50
                #return atomicBlock, analyzeSegment/atomicBlock , NumberOfThreading
        else:  
            print(atomicBlock)
            atomicBlock*=(analyzeSegment/atomicBlock)/50
    NewLenght, NewHeight= int(length*(atomicBlock/(height*length))),int(height*(atomicBlock/(height*length)))
    print('Количество ридеров:',analyzeSegment/atomicBlock)
    print('Измененный атомарный блок равен:',atomicBlock)
    print('Измененные стороны равны', NewLenght, NewHeight)
    return NewLenght,NewHeight, analyzeSegment/atomicBlock, NumberOfThreading


def Edges(image_path):
    image= Image.open(image_path)
    edges=image.filter(ImageFilter.FIND_EDGES)
    return edges

def GettingRidOfInterferences(image):
    #matrix_analyze,matrix=np.asarray(image, dtype='uint8'),[np.asarray(image, dtype='uint8')]
    #plt.imshow(matrix_analyze,interpolation='none')
    #plt.show()
    #column=len(matrix[0])
    #row=len(matrix[0][0])
    #matrix=np.asarray(image, dtype='uint8')
    #matrix3x3=[[1,1,1],[1,1,1],[1,1,1]]    #matrix5x5=[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
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
            else:  matrix_analyze[x][y]=0
    print(x)    """
    #__Рабочая версия с наибольшим качеством отрисовки__
    """img = plt.imread('F:\Python\Dataset-original\\2.bmp')
    plt.imshow(img,interpolation='none')
    plt.show()
    img_bw = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')


    kernel_open =cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    kernel_close =cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    masking=cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel_close)
    masking = cv2.morphologyEx(masking, cv2.MORPH_OPEN, kernel_open)
   
    masking=cv2.dilate(masking, kernel_open, iterations=1)

    masking=cv2.morphologyEx(masking, cv2.MORPH_GRADIENT, kernel_open)
    
    masking=cv2.dilate(masking, kernel_open, iterations=2)

    plt.imshow(masking,interpolation='none')
    plt.show()"""

    img = plt.imread('D:\Python\Dataset-original\\2.bmp')
    plt.imshow(img,interpolation='none')
    #plt.show()
    img_bw = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype('uint8')

    kernel_open =cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    kernel_close =cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    masking=cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel_close)
    masking = cv2.morphologyEx(masking, cv2.MORPH_OPEN, kernel_open)
   
    masking=cv2.dilate(masking, kernel_open, iterations=1)

    masking=cv2.morphologyEx(masking, cv2.MORPH_GRADIENT, kernel_open)
    
    masking=cv2.dilate(masking, kernel_open, iterations=2)

    kernel_open =cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    kernel_close =cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    masking=cv2.morphologyEx(masking, cv2.MORPH_CLOSE, kernel_close)
    masking = cv2.morphologyEx(masking, cv2.MORPH_OPEN, kernel_open)

    print('Размеры изображения')
    AnalyzeArrayLenght=np.size(masking[0])
    AnalyzeArrayHeight=np.size(masking)/np.size(masking[0]) 
    #print(np.size(masking[0][0:]),np.size(masking[0]))
    #test.LBPlearning(masking,1)

    #print(np.size(masking[0]),int((np.size(masking)/np.size(masking[0]))))
    
    #print(separator(np.size(masking[0]),int((np.size(masking)/np.size(masking[0])))))
    
    PostSeparator=Separator(np.size(masking[0]),int((np.size(masking)/np.size(masking[0]))))
    
    print(PostSeparator)
    
    PostSpeedcheck=SpeedCheck(AnalyzeArrayLenght,AnalyzeArrayHeight, PostSeparator[0],PostSeparator[1])
    
    #print(PostSpeedcheck[0],PostSpeedcheck[1])

    

    plt.imshow(masking/128,interpolation='none')
    #plt.show()

    #CreateMatrix(PostSpeedcheck[1],PostSeparator[0],PostSpeedcheck[1])

    #LBP()

    #CreateTreading(PostSeparator[1])


"""
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

    mask = np.dstack([mask, mask, mask]) / 255
    out = img * mask

    plt.imshow(out,interpolation='none')
    plt.show()"""


'''
    plt.imshow(matrix_analyze,interpolation='none')
    plt.show()
    image=Image.fromarray(matrix_analyze,'RGB')
    image.show()'''


def Normalization(image_path):
    normal_image=GettingRidOfInterferences(Edges(image_path))
 
Normalization(f'D:\Python\Dataset-original\\безымянный011.jpg') 