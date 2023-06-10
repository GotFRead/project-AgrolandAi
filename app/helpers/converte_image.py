
import io
from PIL import Image, ImageDraw
import numpy
from enum import Enum 
import time
import os
import cv2
import matplotlib as plt
class Filters(tuple, Enum):
    FIRST = (0,2,0)
    SECOND = (2,0,0)

def read_image_from_bytearray(bytearray):
    image = Image.open(io.BytesIO(bytearray)).convert('RGB')
    name = f"{time.time_ns()}.tiff"
    image.save(name, 'TIFF')
    return Image.open(name)


def open_image(path_to_image):
    return Image.open(path_to_image)

def convert_image_to_array(image):
    return numpy.asarray(image).astype('int32')

def get_filter(object, result):
    img = Image.fromarray((object * 255).astype(numpy.uint8))
    red, green, blue = img.split()
    zeroed_band = red.point(lambda _: 0)
    if result == Filters.SECOND:
        tpl_res = (red, zeroed_band, zeroed_band)
    else:
        tpl_res = (zeroed_band, green, zeroed_band)
    image = Image.merge("RGB",tpl_res)
    return numpy.asarray(image).astype('int32')
    
def convert_array_to_image(object):
    name_image = f"{time.time_ns()}.png"
    img_float32 = numpy.float32(object)
    img_float32 = cv2.cvtColor(img_float32, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name_image, img_float32)
    return name_image

def delete_temp_image(name_image):
    os.remove(name_image)

def resize_image(object, shape):
    img_float32 = numpy.float32(object)
    return cv2.resize(img_float32, shape)

def create_temp_image(object):
    name_image = f"{time.time_ns()}.tiff"
    Image.SAVE(name_image, object)
    return name_image

def create_filter(object, result):
    img_float32 = object.astype(numpy.uint8)
    img_float32 = cv2.cvtColor(img_float32, cv2.COLOR_RGB2RGBA)
    
    if result == Filters.SECOND:
        result = (255, 0, 0)
    elif result == Filters.FIRST:  
        result = (0, 255, 0)

    img = Image.fromarray(img_float32)
    img.convert("RGBA")

    draw = ImageDraw.Draw(img)
    TRANSPARENCY = .25  # Degree of transparency, 0-100%
    OPACITY = int(255 * TRANSPARENCY)
    TINT_COLOR = result

    overlay = Image.new('RGBA', img.size, TINT_COLOR+(0,))
    draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
    draw.rectangle(((0, 0), (1000, 1000)), fill=TINT_COLOR+(OPACITY,))
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    # img.show()
    return img

def change_pixels(object):
    for i in range(len(object)):
        for j in range(len(object[0])):
            # img[i, j] is the RGB pixel at position (i, j)
            # check if it's [0, 0, 0] and replace with [255, 255, 255] if so
            if object[i, j].sum() < 0:
                object[i, j] = [255, 255, 255]
            else:
                object[i, j] = [0, 0, 0]
    return object

if __name__ == '__main__':
    # read_image_from_bytearray()
    # a = [1, 151, 1

    img = Image.new("RGBA", (1000,1000))
    draw = ImageDraw.Draw(img)

    draw.rectangle((1000,1000,500,300), fill=(1000,1000,255,10), 
                outline="black", width=2)
    img.show()
    # b = [[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]] * [1,0,0]
    # # b[0] = numpy.sum([1, 151, 1], b[0])
    # # b[0] = a
    # print(b)