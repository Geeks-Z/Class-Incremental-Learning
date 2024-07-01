'''
Descripttion:
version: 1.0
Author: hwzhao
Date: 2022-10-13 09:19:09
'''
# 引入模块
#!/usr/bin/env python
# coding: utf-8
import os
import sys
from PIL import Image


f = open(r"/gs/home/izhw1024/Code/ECCV22-FOSTER-master/imagenet-sub/eval.txt",encoding='utf-8')
lines = f.readlines()  # 读取全部内容
for line in lines:
    line_array = line.split('\n')
    if(line_array[0].endswith("JPEG")):

        line_array = line.split('\n')  # 拆分字符串
        line_array2 = line_array[0].split('/')  # 拆分字符串
        first_col = line_array2[0]  # 分割出文件夹名称
        second_col = line_array2[1]  # 分割出图片名称
        # second_col = second_col.replace('\\', '/')  # 转换字符

        moveToPath = (r"/gs/home/izhw1024/Code/ECCV22-FOSTER-master/data/imagenet100/val/%s" %
                      (first_col))  # 定义要移动的目标目录
        target_ImageName = line_array2[1]  # 目标图像名称
        moveToPathImage = os.path.join(moveToPath, target_ImageName)  # 合并字符串
        moveFromPathBasic = os.path.join("/gs/home/izhw1024/Dataset/imagenet/val", first_col)
        moveFromPath = os.path.join(moveFromPathBasic, second_col) # 定义要移动的图像路径
        # moveFromPath = moveFromPath.strip()  # 去除首位空格
        # moveToPathImage = moveToPathImage.strip()  # 去除首位空格

        if(os.path.exists(moveFromPath)):  # 文件存在
            # 复制图片到新文件夹
            img = Image.open(str(moveFromPath))
            img.save(moveToPathImage)
            # os.remove(str(moveFromPath))  # 删除图片
        else:
            print("不存在", moveFromPath)
        # a = 0
        # if(i % 100 == 0):
        #     print("当前序号%d" % i)
            # print("原路径：",moveFromPath)
            # print("目标路径：", moveToPathImage)
    else:
        continue

print(len(lines))
