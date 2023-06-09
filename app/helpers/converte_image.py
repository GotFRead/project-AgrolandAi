
import io
import PIL.Image as Image


def read_image_from_bytearray(bytearray):
    return Image.open(io.BytesIO(bytearray))

if __name__ == '__main__': 
    read_image_from_bytearray()