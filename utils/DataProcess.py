import os
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


def rename_fivek(dir):
    fileList = os.listdir(dir)
    for file in fileList:
        if file[0] == 'a':
            file_s = file.split('-')
            newName = file_s[0].strip('a')
            os.rename(os.path.join(dir, file), os.path.join(dir, newName + '.jpg'))


def read_img(file_name):
    img = tf.io.read_file(file_name)
    img = tf.io.decode_image(img, dtype=tf.float32)
    return img


def showImg(img):
    plt.imshow(tf.squeeze(img))
    plt.axis('off')
    plt.show()


def resize_single_img(file_name, size):
    img = Image.open(file_name)
    img = img.resize(size)
    img.save(file_name)


def resize_batch_img(dir, size):
    fileList = os.listdir(dir)
    for file in fileList:
        fileName = os.path.join(dir, file)
        resize_single_img(fileName, size)


def process_data(original_X_path, original_Y_path, target_X_path,target_Y_path, num, resize=None):
    randList = np.random.randint(1, 5000, num)
    if not os.path.exists(target_X_path):
        os.mkdir(target_X_path)
    if not os.path.exists(target_Y_path):
        os.mkdir(target_Y_path)
    for i in randList:
        fileName = f'%04d' % i + '.jpg'
        shutil.copyfile(os.path.join(original_X_path, fileName), os.path.join(target_X_path, fileName))
        shutil.copyfile(os.path.join(original_Y_path, fileName), os.path.join(target_Y_path, fileName))
    if resize:
        resize_batch_img(target_X_path, resize)
        resize_batch_img(target_Y_path, resize)


def geneDataSet(dir_X, dir_Y, batch_size):
    fileList_X = os.listdir(dir_X)
    X_list = []
    Y_list = []
    for file in fileList_X:
        fileNameX = os.path.join(dir_X, file)
        imgX = read_img(fileNameX)
        X_list.append(imgX)
        fileNameY = os.path.join(dir_Y, file)
        imgY = read_img(fileNameY)
        Y_list.append(imgY)
    train_data = (X_list, Y_list)
    return tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)