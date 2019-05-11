import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import inspect
import importlib
from cv2 import HOGDescriptor
from skimage.feature import hog 
import pickle

#HOGDescriptor
cv2_hog = cv2.HOGDescriptor("hog.xml")

def draw(img):
    plt.imshow(img, cmap = 'gray')
    plt.show()

def cvt_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

