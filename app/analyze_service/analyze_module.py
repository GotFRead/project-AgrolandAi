
import numpy as np
import tensorflow as tf
import cv2
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets, layers,  models
import sys
sys.path.insert(1, r'D:\Python\Project_PVS-CNN\app\helpers')
import Morfology_updating_image as morfy
import Converte_image
import main

class Analyzer:
    def __init__(self, path_to_module) -> None:
        self.model = load_model(path_to_module)

    def identification(self, object):
        x = np.expand_dims(object, axis=0)
        prediction = self.model.predict(x)
        result = int(prediction[0][0])
        return class_names[result]


logger = main.create_logger("analyze_module")

model = Analyzer(r"C:\Users\1\Downloads\prototype_CNN_1_4_val_acc_99.h5")


def init_config(path_to_config):
    with open(path_to_config, 'r') as save:
        config = json.load(save)
        if config is not None:
            return config
        else:
            raise FileNotFoundError


config = init_config(
    r'D:\Python\Project_PVS-CNN\app\analyze_service\configs\analyze_module.json')


class Worker:
    """ Exec unit """

    def __init__(self, object) -> None:
        self.config = config['worker']

    async def preparing_image(self, image):
        return cv2.resize(image, (x for x in self.config['image_format']))

    async def analyze(image):
        return model.identification(image)


class solutionFactory:
    def __init__(self, content, params) -> None:
        logger.info("In solution factory")
        self.source_object = Converte_image.read_image_from_bytearray(content)
        self.params = params
        self.config = config['solutionFactory']
        self.creating_appropriate_matrix()

    def creating_appropriate_matrix(self):
        if 'atomic_block' in self.config:
            blocks_in_source = [
                self.source_object.size[0] / self.config['atomic_block'][0],
                self.source_object.size[1] / self.config['atomic_block'][1]
            ]

            for x in range(len(blocks_in_source)):
                block = blocks_in_source[x]
                if (block/0.1) % 10 != 0:
                    block = block + ((block/0.1) - (block/0.1) % 10)
                    # if self.source_object.size[0] % self.config['atomic_block'][0]:
                blocks_in_source[x] = int(block)
            pixel = [0,0,0]
            appropriate_matrix = [[pixel] * blocks_in_source[1] * self.config['atomic_block'][0]] * blocks_in_source[0] * self.config['atomic_block'][1]
            return appropriate_matrix
        else:
            logger.error("Config invalid field 'atomic_block' not found")

    def division_source_object(self):
        pass

    async def create_worker(self):
        pass


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
        # print(result)
        return class_names[result]

    img = __analyze(
        r"D:\Python\DataFrame\rgb_channel_images\test_validate\склон\601912.png")
    print(f'Тип агроландшафта: {img}')
