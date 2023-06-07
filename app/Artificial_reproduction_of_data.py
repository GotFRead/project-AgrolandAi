from PIL import Image
import os

def rotate_image(name_image, angle_rotate):
    image = Image.open(name_image)
    rotated_image = image.rotate(angle_rotate)
    name, format_image = name_image.split('.') 
    rotated_image.save(f'{name}_{angle_rotate}.{format_image}')
    image.close()

def create_dublicate(name_image):
    for angle in range(90,271,90):
        rotate_image(name_image, angle)

def rotate_image_to_dir(path_to_dir):
    for root, dirs, files in os.walk(path_to_dir, topdown=False):
        for file in files:
            path_to_object = path_to_dir + '\\'+ file
            create_dublicate(path_to_object)
            # delete_image_to_name(path_to_object)


def delete_image_to_name(path_to_object):
    name = path_to_object.split('\\')[-1]
    print(len(name.split('_')))
    if len(name.split('_')) != 1:
        os.remove(path_to_object) 

if __name__ == '__main__':
    rotate_image_to_dir('D:\Python\Project_PVS-CNN\равнина')