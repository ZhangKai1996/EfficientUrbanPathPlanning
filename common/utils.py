import math

import numpy as np


def softmax(x):
    """
    x: 输入的向量或矩阵（支持 batch）
    return: softmax 输出，形状与 x 相同
    """
    # 如果是二维的，按行做 softmax
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)  # 避免数值溢出
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        x = x - np.max(x)  # 避免数值溢出
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)


def ont_hot(idx, size):
    array = np.zeros(shape=size)
    array[idx] = 1.0
    return array


def haversine_coord(p1, p2, km=False):
    lat1, lon1 = p1
    lat2, lon2 = p2
    # and longitudes
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0
    # convert to radians
    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0
    # apply formulae
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) * math.cos(lat1) * math.cos(lat2))
    c = 2 * math.asin(math.sqrt(a))
    if km: return 6371 * c * 1000.0
    return 6371 * c


def haversine(p1, p2, km=False):
    return haversine_coord(p1=(p1['y'], p1['x']), p2=(p2['y'], p2['x']), km=km)


def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p1-p2, 2)))
