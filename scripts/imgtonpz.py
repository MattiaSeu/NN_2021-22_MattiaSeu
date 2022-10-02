from PIL import Image
import numpy as np
import os, sys

path = "C:/Users/DaPC/PycharmProjects/ldm1/data/ref_datasets/COCO/val2017"
dirs = os.listdir(path)

#
# def resize():
#     for item in dirs:
#         if os.path.isfile(path + item):
#             im = Image.open(path + item)
#             f, e = os.path.splitext(path + item)
#             imResize = im.resize((256, 256), Image.ANTIALIAS)
#             imResize.save(f + '.jpg', 'jpg', quality=80)
#
#
# resize()


path_to_files = "C:/Users/DaPC/Desktop/University/NeuralNetworks/datasets/ffhq/resized/"
array_of_images = []

for _, file in enumerate(os.listdir(path_to_files)):
    single_im = Image.open(path_to_files + file)
    single_array = np.array(single_im)
    array_of_images.append(single_array)

np.savez(path_to_files + "all_images.npz", array_of_images)  # save all in one file

