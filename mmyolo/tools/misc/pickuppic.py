# Jiaxiong Yang reserved.
# 编写的工具，专门用于把不同模型得到的结果图片部分（以path里面的图片为准）拷贝到对应文件夹下，方便可视化对比
import os
import shutil

path = './'
d_1 = './yolov6/'
d_2 = './yolov8/'
d_3 = './ppyoloe/'
d_4 = './ours/'
save_1 = './yolov6_pick/'
save_2 = './yolov8_pick/'
save_3 = './ppyoloe_pick/'
save_4 = './ours_pick/'

d = [d_1, d_2, d_3, d_4]
save_d = [save_1, save_2, save_3, save_4]

file_list = os.listdir(path)
for file in file_list:
    if file.endswith('.jpg'):
        for d, save in zip(d, save_d):
            if os.path.exists(d + file):
                shutil.copy(d + file, save)
