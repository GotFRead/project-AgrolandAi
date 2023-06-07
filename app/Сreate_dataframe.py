import pandas as pd
import numpy
import math
import os
import matplotlib.pyplot as plt
import pprint
import json
import cv2
import Morfology_updating_image as morfy
import tarfile
import zipfile
import shutil
from enum import Enum

class TypeOfArchiving(str, Enum):
    DntArchive  = 0
    Archive     = 2

class CreateDataframe:
    DEFAULT_NAME_TEMP_DIR = 'temp_dir'

    def __init__(self, path_to_object, path_to_save, dataframe_name, batch_size, type_archive) -> None:
        self.batch_size = batch_size
        self.path_to_save = path_to_save
        self.path_to_object = path_to_object
        self.dataframe_name = dataframe_name

        self.list_daily_files = list()
        self.files_list = list()
        self.categories_list = list()
        self.values_list = list()
        self.dataframe = dict()
        self.arch_flag = type_archive
        self.path_to_arch_dir = str()

        self.create_temp_dir()
        self.create_temp_dataframe()
        self.load_temp_result()
        # self.create_temp_arch()
        # self.archive()


    # def __del__(self):
    #     pass

    def create_temp_dir(self):
        os.chdir(self.path_to_object)
        if CreateDataframe.DEFAULT_NAME_TEMP_DIR not in os.listdir(self.path_to_save):
            os.mkdir(self.path_to_save +
                     f'\{CreateDataframe.DEFAULT_NAME_TEMP_DIR}')

    def create_temp_dataframe(self):
        files_list = list()
        categories_list = list()
        for root, dirs, files in os.walk(os.getcwd(), topdown=True):
            for file in range(0, len(files)):
                if file % self.batch_size == 0:
                    temp_data = dict()
                    temp_data['files'] = files_list
                    temp_data['categories'] = categories_list

                    frame = pd.DataFrame(temp_data)
                    frame.to_csv(
                        self.path_to_save + f'\{CreateDataframe.DEFAULT_NAME_TEMP_DIR}' + f'\\temp_file_{file}.csv', index=False)

                    files_list = list()
                    categories_list = list()

                if files[file].split('.')[-1] in ['png', 'jpeg']:
                    morfy.convert_to_channel_one_channel(
                        root + f'/{files[file]}')
                    image = plt.imread(root + f'/{files[file]}')
                    arrays = numpy.asarray(image, dtype='uint8')
                    files_list.append(arrays.tolist())
                    categories_list.append(root.split('\\')[-1])

    def clear_info(self, tar_info):
        tar_info.uid = tar_info.gid = 0
        tar_info.uname = tar_info.gname = 'root'

    def load_temp_result(self):
        res_file = open(f"{self.dataframe_name}.csv", 'a')

        res_file.writelines(',files,category\n')

        for root, dirs, files in os.walk(self.path_to_save + f'\{CreateDataframe.DEFAULT_NAME_TEMP_DIR}'):
            for number_file in range(len(files)):
                full_size = len(files) * self.batch_size
                with open(root + '\\' + files[number_file], 'r') as file:         
                    content = file.read()
                    content = content.split(""" "[""".replace(' ', ''))
                    
                    for number_object in range(1, len(content)):
                        image, category = content[number_object].split("""]", """.replace(' ', ''))
                        result_string = f"""{number_file * self.batch_size + number_object},\
                            {image},\
                            {category}\n"""
                        res_file.writelines(result_string)
                        print(f"Complete {((number_file * self.batch_size + number_object)/full_size) * 100}%")
        
        self.create_temp_arch()
        self.archive()
        res_file.close()
                        
    def create_dataframe(self, files, dataframe_name):
        self.dataframe['files'] = files_list
        self.dataframe['categories'] = categories_list

        frame = pd.DataFrame(self.dataframe)
        frame.to_csv(f'{dataframe_name}.csv', index=False)

    def create_temp_arch(self):
        self.path_to_arch_dir = self.path_to_save + f'/temp_arch_dir'
        if CreateDataframe.DEFAULT_NAME_TEMP_DIR not in os.listdir(self.path_to_save):
            os.mkdir(self.path_to_arch_dir)

    def archive(self):
        if self.arch_flag == TypeOfArchiving.Archive:
            shutil.make_archive(self.dataframe_name, 'zip', self.path_to_arch_dir)
                
        elif self.arch_flag == TypeOfArchiving.DntArchive:
            return None
        
        # if self.arch_flag == TypeOfArchiving.Archive:
        #     file = tarfile.open(f"{self.dataframe_name}.tar", "w:gz")
        #     file.add(f"{self.dataframe_name}.csv")
        # elif self.arch_flag == TypeOfArchiving.DntArchive:
        #     return None




if __name__ == '__main__':
    # CreateDataframe(r'D:\Python\DataFrame\one_channel_images\train',
    #                 os.getcwd(), 'dataframe_train', 25, TypeOfArchiving.Archive)

    # with tarfile.open("D:/Python/Project_PVS-CNN/dataframe_train.tar", 'r:gz') as file:
    #     # print(file.read())
    #     file.extractall()

    # shutil.make_archive('arch', 'zip', r"D:\Python\Project_PVS-CNN\arch_test")

    # with zipfile.ZipFile('arch.zip') as arch_zip:
    #     with arch_zip.open("dataframe_test_validate.csv") as csv_file:
    #         print(csv_file.read())

    # with zipfile.ZipFile ('Python.zip', 'w', zipfile.ZIP_DEFLATED) as zip_file:
    #     zipfile()

    # with open("D:\Python\Project_PVS-CNN\dataframe_test_validate.csv") as file:
    #     file.read()

    with zipfile.ZipFile("D:\Python\DataFrame\one_channel_frame\dataframe_train.zip", 'r') as arch:
        for file_name in arch.namelist():
            with open(file_name) as file:
                file.read()

    pass