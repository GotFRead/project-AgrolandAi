from datetime import datetime
from time import time_ns
from turtle import pos, position, window_width
from unittest import result
import pygame
import os
import io
from PIL import Image, ImageTk
import PySimpleGUI as py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pygame.locals import *
from tkinter import messagebox
from win32api import GetSystemMetrics
import logging

def cleaner_log():
    current_path = os.getcwd()
    for root, dirs, files in os.walk(current_path, topdown=False):
        for name in files:
            if name == 'log_file.txt':
                os.remove(f'{current_path}\log_file.txt')


cleaner_log()

logger = logging.getLogger('main_application')
logger.setLevel(logging.INFO)

file_logger = logging.FileHandler('log_file.txt')


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
file_logger.setFormatter(formatter)

logger.addHandler(file_logger)
logger.info('Start application')


def open_image(path_image: str) -> object :
    image = plt.imread(f'{path_image}')

    logger.info(f'Открытие изображения {path_image}')

    plt.imshow(image)
    plt.show()


def back_to_root_dirs() -> None:
    old_path = os.getcwd()
    #print('Старый путь :'+ old_path)
    new_path = str()
    part_path = old_path.split('\\')
    for part in range(0,len(part_path)-1):
        new_path+=part_path[part]
        new_path+='\\'
    
    logger.info(f'Переход в корневую папку из {old_path} , в {new_path}')

    #print('Новый путь :'+ new_path)
    os.chdir(f'{new_path}')
    

def info_of_image(image: object) -> int:
    matrix_analyze,matrix=np.asarray(image, dtype='uint8'),[np.asarray(image, dtype='uint8')]
    #plt.imshow(matrix_analyze,interpolation='none')
    #plt.show()
    column=len(matrix[0])
    row=len(matrix[0][0])
    logger.info(f'Информация об изображении: высота {row}, ширина {column}')
    return row, column


def create_dataset_object(screen , position: tuple ,size=(1000,1000), name= 'datasets__objects.jpg'):
    object_dataset = pygame.Surface(size)
    object_dataset.blit(screen, (0,0) ,(position, size))

    pygame.image.save(object_dataset, name)

    logger.info(f'Изображение - {name} - сохранено')

def create_dir(name:str) -> None:
    if name == None:
        logger.error(f'Папка не создана!')
        return False
    path = os.getcwd()
    logger.info(f'Папка - {name} - создана')
    os.mkdir(f'{path}/{name}')


def change_dir(name:str) -> None:
    path = os.getcwd()

    path +=f'\{name}'

    logger.info(f'Переход в папку {path}')

    #path.join(name)  
    os.chdir(path)
    #print(os.getcwd())


def naming_dirs():
    #Create layout
    layout2 = [[py.Text('Введите название новой папки')],[py.Input(),],[py.Ok()] ]

    window = py.Window('Название папки', layout2)

    event , values = window.read()

    logger.info(f'Папка выбрана {values[0]}')

    #print(values[0])

    window.close()

    return values[0] 



def naming_image():

    #Create layout
    layout2 = [[py.Text('Введите название файла')],[py.Input(),],[py.Ok()] ]

    window = py.Window('Название файла', layout2)

    event , values = window.read()

    logger.info(f'Файл выбран {values[0]}')

    window.close()

    return values[0] 


def auto_naming_image():
    number_files = 0

    results = ''

    #Create layout
    layout2 = [[py.Text('Введите название папки')],[py.Input(),],[py.Ok()] ]

    window = py.Window('Название файла', layout2)

    event , values = window.read()

    name_dirs = values[0]

    window.close()

    current_path = os.getcwd()
    for root, dirs, files in os.walk(current_path, topdown=False):
        if dirs == f'{name_dirs}':
            for name in files:
                number_files +=1

    if number_files ==0:
        return f'{name_dirs}-1'

    result = [f'{name_dirs}-{number_files}', name_dirs]

    return result
    
def create_background_image(path_image: str, scale_percent = 0):
    image = plt.imread(f'{path_image}')

    if scale_percent == 0:
        scale_percent = auto_scale_background(image.shape[0], image.shape[1])
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

    else:
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    image_background = 255*(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)).astype('uint8')

    image = 255*(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).astype('uint8')
    #plt.imshow(resized)
    #plt.show()

    logger.info(f'Создание изображения для создания датасетов - {path_image}')


    return image, image_background , scale_percent


def auto_scale_background(row, column):
    screen_info_widht = GetSystemMetrics(0)
    screen_info_height = GetSystemMetrics(1)
    print(screen_info_height, screen_info_widht)
    print(row, column)
    
    if row >= screen_info_height or column >= screen_info_widht:
        result_row  = 100/(row/screen_info_height)
        result_column = 100/(column / screen_info_widht)
        result_scale = (result_column+result_row)/4
    
    else:
        result_scale = 100

    logger.info(f'Результат масштабирования - {result_scale}')

    return result_scale


def create_mapping(path_image: str, scale_percent = 0):

    orginal_image, background_image, scale_percent = create_background_image(path_image = path_image, scale_percent = scale_percent)

    #Создание изоображения для отображения картинки в background
    cv2.imwrite('execute.jpg', img=background_image*255)

    #Создание изоображения для отображения картинки в beckground
    cv2.imwrite('execute_true_scale.jpg', img=orginal_image*255)

    image_true_scale = pygame.image.load("execute_true_scale.jpg")

    background = pygame.image.load("execute.jpg")
    
    row, column =  info_of_image(background_image)    

    logger.info(f'Размер созданной разметки высота {row}, ширина {column}')

    background_rect = background.get_rect(bottomright=(row, column))

    screen = pygame.display.set_mode(size= (row, column))    
    
    screen.fill((100, 150, 200))

    screen.blit(background,background_rect)   

    pygame.display.update()

    conditions = True 
    while conditions == True:      
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                conditions = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:

                    name_image = choose_dirs()
                    if name_image == None :
                        delete_execute_images()
                        logger.info(f'Удаление исполняемых изображений')
                        continue
                    #change_dir()
                    # size_square отвечает за длину и высоту квадрата 
                    size_square = 500                    
                    create_dataset_object(image_true_scale ,\
                        size=(size_square,size_square),\
                        position= (int(event.pos[0]*100/scale_percent) - size_square/2 ,int(event.pos[1]*100/scale_percent) - size_square/2),\
                        name=f'{name_image}.png')
                    back_to_root_dirs()
                    logger.info(f'Размер квадрата вырезки равен : {int(size_square/2*scale_percent/100)}')
                    pygame.draw.rect(screen, 
                    (0, 0, 255), 
                    (event.pos[0] - int(size_square/2*scale_percent/100) , event.pos[1] - int(size_square/2*scale_percent/100) , int(size_square*scale_percent/100) , int(size_square*scale_percent/100)))
                    pygame.display.update()
                    delete_execute_images()
    pygame.quit()
                

def delete_execute_images():
    current_path = os.getcwd()
    for root, dirs, files in os.walk(current_path, topdown=False):
        for name in files:
            if name == 'execute.jpg':
                logger.info(f'Файл по адресу удален: ' + os.path.join(root, name))
                os.remove(f'{current_path}\execute.jpg')
            elif name == 'execute_true_scale.jpg':
                logger.info(f'Файл по адресу удален: ' + os.path.join(root, name))
                os.remove(f'{current_path}\execute_true_scale.jpg')


def auto_naming_image_for_dirs(name_dirs):
    dt = datetime.now()
    time = dt.time()
    time_str = time.__str__()
    time_now = time_str.split(':')
    results=str()
    for times in time_now:
        results += times
    
    
    logger.info(f'Процедурное название объекта: {name_dirs}-{results}')
    return f'{name_dirs}-{results}'


def auto_creater_dirs(string_input: str):
    split_string = string_input.split('-')
    count_dirs, name_category = int(split_string[1]), split_string[0]
    for iterator in range(1,count_dirs+1):   
        try:
            create_dir(f'{name_category}-{iterator}')
        except FileExistsError:
            continue


def auto_create_dirs():
    layout2 = [[py.Text('Введите: Название новой категорий-количество классов категорий ')],[py.Input(),],[py.Ok()] ]

    window = py.Window('Авто создание папок', layout2)

    event , values = window.read()

    logger.info(f'Автоматическлое создание папок {values[0]}')

    #print(values[0])

    window.close()

    return auto_creater_dirs(values[0])



def choose_dirs():
    folder = os.getcwd()

    flist0 = os.listdir(folder)

    list_dir = [dirs for dirs in flist0 if os.path.isdir(s=dirs)]

    count_dirs = len(list_dir) # number of images found



    col_files = [[py.Listbox(values=list_dir, change_submits=True, size=(60, 30), key='listbox')],
                [py.Button('Выбрать папку', size=(8, 2)), py.Button('Назад', size=(8, 2))]]


    layout = [col_files]   

    window = py.Window('Gui-Test', layout, return_keyboard_events=True,
                right_click_menu=py.MENU_RIGHT_CLICK_EDITME_VER_EXIT, finalize=True, 
                location=(0, 0), use_default_focus=False)

    window["listbox"].bind('<Double-Button-1>' , "+-double click-")

    try:
        while True:
        # read the form
            event, values = window.read()
            print(event, values)

            i = 0

            # perform button and keyboard operations
            if event == py.WIN_CLOSED:
                window.close()
                return None
            elif event in ('Далее', 'MouseWheel:Down', 'Down:40', 'Next:34'):
                i += 1
                if i >= count_dirs:
                    i -= count_dirs
                filename = os.path.join(folder, list_dir[i])
            elif event in ('Предыдущие', 'MouseWheel:Up', 'Up:38', 'Prior:33'):
                i -= 1
                if i < 0:
                    i = count_dirs + i
                filename = os.path.join(folder, list_dir[i])
            elif event in ('Выбрать папку'):
                name_dirs = values["listbox"][0]  
                change_dir(name_dirs)
                window.close()
                return auto_naming_image_for_dirs(name_dirs)            
            elif event in ('Назад'):
                window.close()
                return None
            elif event == 'listbox':                # something from the listbox
                f = values["listbox"][0]            # selected filename
                filename = os.path.join(folder, f)  # read this file
                i = list_dir.index(f)                 # update running index         
            elif event == 'listbox+-double click-':
                name_dirs = values["listbox"][0]  
                change_dir(name_dirs)
                window.close()
                return auto_naming_image_for_dirs(name_dirs) 
            else:
                filename = os.path.join(folder, list_dir[i])

    except IndexError as error:
        logger.error(f'Выявлена ошибка {error}')
        choose_dirs()


def main():
    folder = py.popup_get_folder('Image folder to open', default_path='')
    
    if not folder:
        py.popup_cancel('Прерывание')
        raise SystemExit()

    img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp", '.tif')

    flist0 = os.listdir(folder)


    fnames = [f for f in flist0 if os.path.isfile(
        os.path.join(folder, f)) and f.lower().endswith(img_types)]

    num_files = len(fnames) # number of images found
    if num_files == 0:
        py.popup('Нет подходящих файлов')
        logger.error('Подходящие файлы не обнаружены')
        raise SystemExit()

    # ------------------------------------------------------------------------------
    # use PIL to read data of one image
    # ------------------------------------------------------------------------------

    def get_img_data(f, maxsize=(1200, 850), first=False):
        """Generate image data using PIL
        """
        img = Image.open(f)
        img.thumbnail(maxsize)
        if first: # tkinter is inactive the first time
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            del img
            return bio.getvalue()
        return ImageTk.PhotoImage(img)
    # ------------------------------------------------------------------------------

    # make these 2 elements outside the layout as we want to "update" them later
    # initialize to the first file in the list
    filename = os.path.join(folder, fnames[0])  # name of first file in list
    image_elem = py.Image(data=get_img_data(filename, first=True))
    filename_display_elem = py.Text(filename, size=(80, 3))
    file_num_display_elem = py.Text('File 1 of {}'.format(num_files), size=(15, 1))

    # define layout, show and read the form
    col = [[filename_display_elem],
        [image_elem]]

    col_files = [[py.Listbox(values=fnames, change_submits=True, size=(60, 30), key='listbox')],
                [py.Button('Формирование DataSet', size=(12, 2)) ,py.Button('Сформировать папку', size=(12,2)) , file_num_display_elem],
                [py.Button('Автоматическое формирование папок', size=(30,2))],
                [py.Text(size=(40,1), key='-OUTPUT-1')]]
    ''',
    [py.Text(size=(40,1), key='-OUTPUT-2')]'''

    layout = [[py.Column(col_files), py.Column(col)]]

    window = py.Window('Gui-Test', layout, return_keyboard_events=True,
                    location=(0, 0), use_default_focus=False)
    try:
        while True:
        # read the form
            event, values = window.read()
            print(event, values)

            i = 0

            # perform button and keyboard operations
            if event == py.WIN_CLOSED:
                break
            #elif event in ('Далее', 'MouseWheel:Down', 'Down:40', 'Next:34'):
            #    i += 1
            #    if i >= num_files:
            #        i -= num_files
            #    filename = os.path.join(folder, fnames[i])
            #elif event in ('Предыдущие', 'MouseWheel:Up', 'Up:38', 'Prior:33'):
            #    i -= 1
            #    if i < 0:
            #        i = num_files + i
            #    filename = os.path.join(folder, fnames[i])
            elif event in ('Формирование DataSet'):
                f = values["listbox"][0]  
                filename = os.path.join(folder, f)
                create_mapping(filename)              
            elif event in ('Сформировать папку'):
                create_dir(naming_dirs())
            elif event in ('Автоматическое формирование папок'):
                auto_create_dirs()
            elif event == 'listbox':                # something from the listbox
                f = values["listbox"][0]            # selected filename
                filename = os.path.join(folder, f)  # read this file
                i = fnames.index(f)                 # update running index
            else:
                filename = os.path.join(folder, fnames[i])

            # update window with new image
            image_elem.update(data=get_img_data(filename, first=True))
            # update window with filename
            filename_display_elem.update(filename)
            # update page display
            file_num_display_elem.update('File {} of {}'.format(i+1, num_files))
            
        window.close()

    except IndexError as error:
        logger.error(f'Выявлена ошибка {error}')
        window.close()
        main()


if __name__=='__main__':
    main()