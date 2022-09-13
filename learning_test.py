import unittest

from Application_create_test_datasets import * 
import logging

#Рассмотрим тестирование на примере выполнение методов по типу черного ящика
class TestCreaterDatasets(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_info_to_image(self):
        path_image = 'D:\Python\DataSets\безымянный020.jpg'
        image = plt.imread(f'{path_image}')
        self.assertEqual(info_of_image(image), (3307,4677), msg='Выполение провалено, размер найден неверно')


    def test_naming_image(self):
        self.assertEqual(naming_image(),'принято', msg='Выполение провалено, файлу передано неверное имя')


    def test_auto_naming_image_for_dirs(self):
        self.assertEqual(auto_naming_image(),'склон-1-1', msg='Выполение провалено,  файлу передано неверное имя')

def test_log():
    logging.basicConfig(filename="logger.txt", level=logging.INFO)
    
    logging.debug("This is a debug message")
    logging.info("Informational message")
    logging.error("An error has happened!")

if __name__=='__main__':
    
    test_log()
    #unittest.main() 