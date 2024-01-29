# Jiaxiong Yang reserved.

import os
from PIL import Image

root = './VisDrone2019/'
path = 'VisDrone/'

d_1 = 'yolov6/'
d_2 = 'yolov8/'
d_3 = 'ppyoloe/'
d_4 = 'ours/'

d = [d_1, d_2, d_3, d_4]
save_dir = 'jigsaws/'
file_list = os.listdir(os.path.join(root, path))
save_path = os.path.join(root, save_dir)
for file in file_list:
    if file.endswith('.jpg'):
        img = Image.open(os.path.join(root, path, file))
        pic_w = img.width // 2
        pic_h = img.height
        result = Image.new(img.mode, ((len(d) + 1) * pic_w, pic_h))
        result.paste(img.crop((0, 0, pic_w, pic_h)), (0, 0))
        for d_ in d:
            mm = Image.open(os.path.join(root, d_, file))
            result.paste(mm.crop((pic_w, 0, img.width, pic_h)), ((d.index(d_) + 1) * pic_w, 0))
        result.save(save_path)
