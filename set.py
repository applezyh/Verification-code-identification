import cv2 as cv
import numpy as np
import torch
import os
import multiprocessing
import random

root_path = "C:\\Users\\apple\\Desktop\\captcha-master\\captcha-master\\images\\four_digit_test2\\"
dir_list = os.listdir(root_path)
size = int(100)
random.shuffle(dir_list)

def task(args):
    args = args-1
    img = cv.imread(root_path + dir_list[args*size])
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).reshape(1, 1, 60, 160)
    for i in range(args * size + 1, args * size + size):
        imgt = cv.imread(root_path + dir_list[i])
        imgt = cv.cvtColor(imgt, cv.COLOR_BGR2GRAY).reshape(1, 1, 60, 160)
        img = np.append(img, imgt, axis=0)
    return img


def lable_set():
    for i in range(len(dir_list)):
        dir_list[i] = dir_list[i][:4]
        dir_list[i] = list(dir_list[i])
        dir_list[i] = [ord(dir_list[i][0]) - 97, ord(dir_list[i][1]) - 97, ord(dir_list[i][2]) - 97,
                       ord(dir_list[i][3]) - 97]
    re_list = np.array(dir_list)
    return torch.from_numpy(re_list).long()


def dataset():
    # pool = multiprocessing.Pool(8)
    result = task(1)  # pool.map(task, [1, 2, 3, 4, 5, 6, 7, 8])
    re = result
    # for i in range(7):
        # re = np.append(re, result[i + 1], axis=0)
    return (torch.from_numpy(re).float()) / 255
