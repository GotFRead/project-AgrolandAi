import numpy as np 
import PIL as pil 
import cv2 as cv 

#
# Development suspended
#

class TemplateAnalyze(object):
    def __init__(self, row, colomn, NumberReader):
        self.row=row 
        self.colomn=colomn
        self.NumberReader=NumberReader
    def info(self):
        return 'Данные ядер анализа строки: %s, %s'.format(self.row, self.colomn)
def LBPlearning(SquareObj, NumberCalc):
    # Определим локальную область 
    
    print('Запуск обработчика %s',(NumberCalc))

    BinaryPattern=TemplateAnalyze(np.size(SquareObj[0]),int(np.size(SquareObj)/np.size(SquareObj[0])),NumberCalc)

    #countEnd=SquareObj/(BinaryPattern.row *BinaryPattern.colomn)
    #countStart=0
    #count=0
    print(int(np.size(SquareObj)/np.size(SquareObj[0]))-1)
    #Функция превентивного анализа на основе сумм пикселей 

    def Attention(AnalyzeArray):
        sum=np.sum(AnalyzeArray)
        if sum<=0.25 * BinaryPattern.row * BinaryPattern.colomn: 
            return False 
        else: 
            return True
        
    #def Clener(Array):
        # Попробовать методом разрастания проверять смежные сектора на наличие пикселей и при отсутствии удалят их 
        # Сделать проверку и чистку 
        # Реализовать разратание паттернов  

    

    for schetY in range(0, np.size(SquareObj[0])-1):
        for schetX in range(0, int(np.size(SquareObj)/np.size(SquareObj[0]))-1):
            if Attention(SquareObj[schetX:schetX+1][schetY:schetY+1])== True:
                print('Attention')
                #Clener(SquareObj[schetX:schetX+1][schetY:schetY+1])



"""        if object.row*(schetX+1)<= len(SquareObj[0][0]):
            
        elif  object.colomn*(schetY+1) <= len(SquareObj[0]):
"""
        


if __name__=='__main__':
    matrix= LBPlearning(f'D:\Python\Dataset-original\\2.bmp',10)
    matrix.info()
