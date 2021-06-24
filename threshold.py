import datetime
import os as os
import urllib.request

import cv2 as cv
import numpy as np
import requests
from matplotlib import image, pyplot


def imread(url):
    if (url[0:4] == 'http'):
        req = urllib.request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        preImg = cv.imdecode(arr, -1)
        if (preImg.shape.length != 3 and preImg.shape[2] != 4):
            return None
        length = preImg.shape[0] * preImg.shape[1] * 3
        img = np.arange(length).reshape(preImg.shape[0], preImg.shape[1], 3)
        for i in range(preImg.shape[0]):
            for j in range(img.shape[1]):
                imgij = preImg[i][j]
                img[i][j] = [ imgij[0], imgij[1], imgij[2] ]
        img = np.uint8(img)
    else:
        img = cv.imread(url, cv.IMREAD_COLOR)
    return img

def basicThreshold(image, threshold):
    if(not isinstance(image, np.ndarray)):
        raise Exception("image need to be numpy.ndarray")
    if(len(image.shape) != 2):
        raise Exception("image need to be gray scale")
    shape0 = image.shape[0]
    shape1 = image.shape[1]
    output = np.arange(shape0*shape1).reshape(shape0, shape1)
    for i in range(shape0):
        for j in range(shape1):
            if (image[i][j] >= threshold):
                output[i][j] = 255
            else:
                output[i][j] = 0
    output = np.uint8(output)
    return output

def meanAdaptiveThreshold(image, blockSize, c):
    if(not isinstance(image, np.ndarray)):
        raise Exception("image need to be numpy.ndarray")
    if(len(image.shape) != 2):
        raise Exception("image need to be gray scale")
    if(blockSize % 2 == 0):
        raise Exception("blockSize need to be odd")
    shape0 = image.shape[0]
    shape1 = image.shape[1]
    output = np.arange(shape0*shape1).reshape(shape0, shape1)
    for i in range(shape0):
        for j in range(shape1):
            halfBlock = int(blockSize/2)
            iStartPoint = (i - halfBlock) if (i - halfBlock >= 0) else 0
            jStartPoint = (j - halfBlock) if (j - halfBlock >= 0) else 0
            iFinishPoint = (i + halfBlock + 1) if (i + halfBlock + 1 <= shape0) else shape0
            jFinishPoint = (j + halfBlock) + 1 if (j + halfBlock + 1 <= shape1) else shape1
            threshold = np.mean(image[iStartPoint : iFinishPoint , jStartPoint : jFinishPoint]) - c
            output[i][j] = 255 if (image[i][j] > threshold) else 0
    output = np.uint8(output)
    return output
