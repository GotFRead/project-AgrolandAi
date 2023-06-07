
import os
import io
import PIL.Image as Image
from array import array

def read_image_from_bytearray(bytearray):
    return Image.fromarray(bytearray)

# bytes = read_image_from_bytearray(path)
# image = Image.open(io.BytesIO(bytes))
# image.save()


if __name__ == '__main__': 
    read_image_from_bytearray()