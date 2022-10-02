from PIL import Image
import numpy as np
import os, sys

path_to_files = "log_dir/models/ldm/cin256" # change with path to image folder to be compressed to npz
array_of_images = []

for _, file in enumerate(os.listdir(path_to_files)):
    single_im = Image.open(path_to_files + file)
    single_array = np.array(single_im)
    array_of_images.append(single_array)

np.savez(path_to_files + "all_images.npz", array_of_images)  # save all in one npz archive

