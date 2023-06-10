

import numpy as np
import tensorflow as tf
import cv2
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets, layers,  models
from enum import Enum 
import sys
sys.path.insert(1, r'Create_of_dataset\app\helpers')
import Morfology_updating_image as morfy
import Converte_image
import main
import math
import asyncio
import base64
import cv2

logger = main.create_logger("analyze_module")


def init_config(path_to_config):
    with open(path_to_config, 'r') as save:
        config = json.load(save)
        if config is not None:
            return config
        else:
            raise FileNotFoundError


class Analyzer:
    def __init__(self, path_to_module) -> None:
        self.model = load_model(path_to_module)

    def identification(self, object):
        x = np.expand_dims(object, axis=0)
        prediction = self.model.predict(x)
        result = int(prediction[0][0])
        return result


config = init_config(
    r'E:\Python\Diplom\New_arch_app\Create_of_dataset\analyze_service\configs\analyze_module.json')

model = Analyzer(config['analyzer']['path_to_CCN'])


class Worker:
    """ Exec unit """

    def __init__(self, object) -> None:
        self.config = config['worker']
        self.shape = (100,100)
        self.source_object = object
        self.result = int()

    def execution(self):
        self.normalyze_source_image()
        return self.analyze()

    def analyze(self):
        return model.identification(self.source_object)
    
    def normalyze_source_image(self):
        self.source_object = Converte_image.resize_image(self.source_object, self.shape)
        self.source_object = morfy.cleaner_to_value(self.source_object, 0.5)
        self.source_object = morfy.morphological_change_image(self.source_object)
        self.source_object = Converte_image.change_pixels(self.source_object)
        # check_image(self.source_object)

class color_point(tuple, Enum):
    FIRST = (0,2,0)
    SECOND = (2,0,0)


class solutionFactory:
    def __init__(self, content, params):
        logger.info("In solution factory")
        self.__source_object = Converte_image.read_image_from_bytearray(content)
        self.__source_object = Converte_image.convert_image_to_array(self.__source_object)
        self.params = params
        self.config = config['solutionFactory']
        # check_image(self.__source_object)

    def execution(self):
        self.prepare_matrix = self.creating_appropriate_matrix()
        self.matrix_overlay()
        return asyncio.run(self.init_solution())

    def creating_appropriate_matrix(self):
        if 'atomic_block' in self.config:
            blocks_in_source = [
                self.__source_object.shape[0] / self.config['atomic_block'][0],
                self.__source_object.shape[1] / self.config['atomic_block'][1]
            ]

            for x in range(len(blocks_in_source)):
                blocks_in_source[x] = math.ceil(blocks_in_source[x])

            pixel = [0, 0, 0]
            appropriate_matrix = [[pixel] * blocks_in_source[1] * self.config['atomic_block']
                                  [0]] * blocks_in_source[0] * self.config['atomic_block'][1]
            
            return np.array(appropriate_matrix)
        else:
            logger.error("Config invalid field 'atomic_block' not found")

    def matrix_overlay(self):
        self.prepare_matrix[0:len(self.__source_object),0:len(self.__source_object[0])] =  self.__source_object

    async def init_solution(self):
        for row in range(0, len(self.prepare_matrix),  self.config['atomic_block'][0]):
            for column in range(0, len(self.prepare_matrix[0]), self.config['atomic_block'][1]):
                content_for_unit = self.prepare_matrix[
                    row: row + self.config['atomic_block'][0],
                    column: column + self.config['atomic_block'][1]
                    ]

                await self.create_exec_unit(content_for_unit, row, column)
                # await asyncio.sleep(0.1)

        # check_image(self.prepare_matrix)
        return await self.prepare_image_to_send(self.prepare_matrix)
        

    async def create_exec_unit(self, content_for_unit, row, column):
        unit = Worker(content_for_unit)
        if unit.execution() == 0:
            result = color_point.FIRST
        else:
            result = color_point.SECOND
        self.build_result(content_for_unit, result, row, column)


    def build_result(self, object, result, row, column):
        self.prepare_matrix[
            row: row + self.config['atomic_block'][0],
            column: column + self.config['atomic_block'][1]
            ] = Converte_image.create_filter(object, result)
        
    async def prepare_image_to_send(self, object):
        name = Converte_image.convert_array_to_image(object)
        with open(name, 'rb') as file:
            im_b64 = base64.b64encode(file.read())
        # Converte_image.delete_temp_image(name)
        # check_image(object)
        return im_b64


def check_image(image):
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    class_names = ['1-3', '3 и более']

    def __analyze(image):
        image = plt.imread(f'{image}')
        resize_one_channel_image = cv2.resize(image, (100, 100))
        global class_names
        model = load_model(
            r"C:\Users\1\Downloads\prototype_CNN_1_4_val_acc_99.h5")
        x = np.expand_dims(resize_one_channel_image, axis=0)
        prediction = model.predict(x)
        result = int(prediction[0][0])
        return class_names[result]

    img = __analyze(
        r"D:\Python\DataFrame\rgb_channel_images\test_validate\склон\601912.png")
    print(f'Тип агроландшафта: {img}')
