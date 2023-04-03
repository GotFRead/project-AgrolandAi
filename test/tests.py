import unittest
from app.Application_create_test_datasets import * 
import logging
from screeninfo import get_monitors



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


    def test_auto_scale_background(self):
        row, column = 3000, 6000
        scale = auto_scale_background(row, column)
        self.assertEqual(scale,17, msg='Найденный коэф масштaбирования не верен')

        row_scale, column_scale = int(3000 * (scale/100)), int(6000* (scale/100))

        screen_info_widht, screen_info_height = screen_info()
        
        self.assertTrue(row_scale < screen_info_widht and column_scale <screen_info_height, msg='Найденный коэф неверен')


if __name__=='__main__':
    unittest.main()




