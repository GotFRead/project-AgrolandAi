from matplotlib.cbook import flatten
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from PIL import Image 

def LBPlearning(SquareObj,length,height,schet):
    print(length,height)

    print(np.size(SquareObj[0]),np.size(SquareObj)/np.size(SquareObj[0]))
    #OffsetLenght=
    
    
    #OffsetHeight=
    
    #print(OffsetLenght, OffsetHeight)

    #w, h= int(np.size(SquareObj[0])),int(np.size(SquareObj)/np.size(SquareObj[0]))
    LBPtemplate= np.array([[0,0,0],[0,0,0],[0,0,0]])
    multiplication= SquareObj
    #j=flatten(multiplication)


  
    for schetY in range(1, np.size(SquareObj[0])-1-4300):
        for schetX in range(1, int(np.size(SquareObj)/np.size(SquareObj[0]))-3100):
            print("Итераций",schetY)
            ArrayForMulti=SquareObj[schetX-1:schetX+2,schetY-1:schetY+2]
            multiplication[schetX-1:schetX+2,schetY-1:schetY+2]= ArrayForMulti* LBPtemplate 
            #print(multiplication)
    
    #o=np.reshape(multiplication,(np.size(SquareObj[0]),np.size(SquareObj)/np.size(SquareObj[0])))

    #data=(np.resize())

    #multiplication=np.resize((int(np.size(SquareObj[0]))),int(np.size(SquareObj)/np.size(SquareObj[0])))
    
    #multiplication=np.resize(multiplication[0],multiplication[0][0])
    print(multiplication)
    #print(OffsetLenght, OffsetHeight)
    cv2.imwrite('2.jpg',multiplication*255)

    s=cv2.imread('2.jpg')

    plt.imshow(s)
    plt.show()
    
    #print(multiplication)
    #type(multiplication, array_to_img(multiplication))
    #Matrox=я

    #plt.imshow(Matrox, cmap='gray')
    #plt.show


img = plt.imread('D:\Python\Dataset-original\\2.bmp')
img_bw = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype('uint8')
plt.imshow(img_bw)
plt.show()
matrix= np.asarray(img_bw, dtype='uint8')
LBPlearning(matrix, 500, 500,3)

