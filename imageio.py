import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


def readTiff(img_name):
    img = Image.open(img_name)
    img = np.array(img)
    return img


def writeTiff(img_name, img):
    '''img: uint16 type'''
    h, w = img.shape
    img = Image.fromstring('I;16', (w, h), img.tostring())
    img.save(img_name)


def readTiffsInFileFolder(folder_name):
    if not os.path.exists(folder_name):
        print 'file folder "', folder_name, '" does not exists.'
    img_names = os.listdir(folder_name)
    img_count = len(img_names)

    if img_count == 0:
        print 'no image in folder: "', folder_name, '".'
    h, w = readTiff(os.path.join(folder_name, img_names[0])).shape

    img_datas = np.zeros((img_count, h, w), dtype='uint16')
    i = 0
    for img_name in img_names:
        img_datas[i] = readTiff(os.path.join(folder_name, img_name))
        i = i + 1

    return img_datas


def writeTiffsToFileFolder(folder_name, img_datas, prefix='', postfix='.tif'):
    '''img_datas: [img_count, img_height, img_width], uint16 type''' 
    for i in range(0, img_datas.shape[0]):
        img_name = os.path.join(folder_name, prefix + str(i + 1) + postfix)
        writeTiff(img_name, img_datas[i])


def writePng(img_name, img):
    plt.imsave(img_name, img)


def writePngsToFileFolder(folder_name, img_datas, prefix='', postfix='.png'):
    for i in range(0, img_datas.shape[0]):
        img_name = os.path.join(folder_name, prefix + str(i + 1) + postfix)
        writePng(img_name, img_datas[i])


if __name__ == '__main__':
    pass
