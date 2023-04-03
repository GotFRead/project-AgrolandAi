from datetime import datetime
from time import time_ns
from turtle import pos, position
from unittest import result
import pygame
import os
import io
from PIL  import Image, ImageTk
import PySimpleGUI as py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pygame.locals import *
from tkinter import messagebox


def open_image(path_image: str) -> object :
    image = plt.imread(f'{path_image}')
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

    #print('Новый путь :'+ new_path)
    os.chdir(f'{new_path}')
    

def info_of_image(image: object) -> int:
    matrix_analyze,matrix=np.asarray(image, dtype='uint8'),[np.asarray(image, dtype='uint8')]
    #plt.imshow(matrix_analyze,interpolation='none')
    #plt.show()
    column=len(matrix[0])
    row=len(matrix[0][0])

    return row, column


def create_dataset_object(screen , position: tuple ,size=(250,250), name= 'datasets__objects.jpg'):
    object_dataset = pygame.Surface(size)
    object_dataset.blit(screen, (0,0) ,(position, size))

    print('Объест сохранен')

    pygame.image.save(object_dataset, name)


def create_dir(name:str) -> None:
    if name == None:
        print('Ошибка папка не создана')
        return False
    path = os.getcwd()
    os.mkdir(f'{path}/{name}')


def change_dir(name:str) -> None:
    path = os.getcwd()

    path +=f'\{name}'

    #path.join(name)  
    os.chdir(path)
    #print(os.getcwd())


def naming_dirs():
    #Create layout
    layout2 = [[py.Text('Введите название новой папки')],[py.Input(),],[py.Ok()] ]

    window = py.Window('Название папки', layout2)

    event , values = window.read()

    #print(values[0])

    window.close()

    return values[0] 



def naming_image():

    #Create layout
    layout2 = [[py.Text('Введите название файла')],[py.Input(),],[py.Ok()] ]

    window = py.Window('Название файла', layout2)

    event , values = window.read()

    print(values[0])

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
    


def create_mapping(path_image: str):
    pygame.init()

    image = plt.imread(f'{path_image}')

    scale_percent = 20 

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)


    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    #plt.imshow(resized)
    #plt.show()

    img_bw = 255*(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)).astype('uint8')

    row, column =  info_of_image(img_bw)    


    cv2.imwrite('execute.jpg', img=img_bw*255)

    background = pygame.image.load("execute.jpg")

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
                        delete_execute_image()
                        continue
                    #change_dir()                    
                    create_dataset_object(screen , position= (event.pos[0] - 50 ,event.pos[1] - 50), name=f'{name_image}.png')
                    back_to_root_dirs()
                    pygame.draw.rect(screen, 
                    (0, 0, 255), 
                    (event.pos[0] - 50 , event.pos[1] -50 , 250 , 250))
                    pygame.display.update()
                    delete_execute_image()



    pygame.quit()
                

def delete_execute_image():
    current_path = os.getcwd()
    for root, dirs, files in os.walk(current_path, topdown=False):
        for name in files:
            if name == 'execute.jpg':
                print('Файл по адресу удален: ' + os.path.join(root, name))
                os.remove(f'{current_path}\execute.jpg')



def auto_naming_image_for_dirs(name_dirs):
    dt = datetime.now()
    time = dt.time()
    time_str = time.__str__()
    time_now = time_str.split(':')
    results=str()
    for times in time_now:
        results += times
    
    
    print(results)
    return f'{name_dirs}-{results}'



def choose_dirs():
    folder = os.getcwd()

    flist0 = os.listdir(folder)

    list_dir = [dirs for dirs in flist0 if os.path.isdir(s=dirs)]

    count_dirs = len(list_dir) # number of images found



    col_files = [[py.Listbox(values=list_dir, change_submits=True, size=(60, 30), key='listbox')],
                [py.Button('Выбрать папку', size=(8, 2)), py.Button('Назад', size=(8, 2))]]


    layout = [col_files]   

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
                #print(os.getcwd())
                window.close()
                return auto_naming_image_for_dirs(name_dirs)            
            elif event in ('Назад'):
                window.close()
                return None
            elif event == 'listbox':                # something from the listbox
                f = values["listbox"][0]            # selected filename
                filename = os.path.join(folder, f)  # read this file
                i = list_dir.index(f)                 # update running index
            else:
                filename = os.path.join(folder, list_dir[i])

        window.close()

    except IndexError as error:
        print(f'Выявлена ошибка {error}')
        choose_dirs()


def main():

    folder = py.popup_get_folder('Image folder to open', default_path='')
    
    if not folder:
        py.popup_cancel('Прерывание')
        raise SystemExit()

    img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")

    flist0 = os.listdir(folder)


    fnames = [f for f in flist0 if os.path.isfile(
        os.path.join(folder, f)) and f.lower().endswith(img_types)]

    num_files = len(fnames) # number of images found
    if num_files == 0:
        py.popup('Нет подходящих файлов')
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
                [py.Button('Формирование DataSet', size=(12, 2)) ,py.Button('Сформировать папку', size=(12,2)), file_num_display_elem],
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
        print(f'Выявлена ошибка {error}')
        window.close()
        main()


if __name__=='__main__':
    main()