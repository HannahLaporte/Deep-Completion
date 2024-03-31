import os
import numpy as np 
from PIL import Image

save_dir = '/root/autodl-tmp/VA-DepthNet/Dataset/save/'
sub_dirs = ['epoch4']

n = 0
for sub_dir in sub_dirs:
    path = os.path.join(save_dir, sub_dir)
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.npy'):
                npy_path = os.path.join(root, file)
                data = np.load(npy_path)
                data = data.reshape(352, 1216) * 256
                png_path = os.path.join(root, 'png', file.replace('.npy', ''))
                img = Image.fromarray(data.astype(np.uint16))
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img.save(png_path)
            print(png_path)

            # if n == 0:
            #     break