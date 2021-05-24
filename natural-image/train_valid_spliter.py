from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



def train_val_spliter(dirname, foldername, classes, rate=0.7, val_rate=0.9):
  count = 0
  datapath = os.getcwd()
  i = 0
  for (path, dic, files) in os.walk(dirname):
    if path.split('\\')[-1] == classes:
      number = len(files)
      for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.jpg':
          if count < number * rate:
            image = cv2.imread(dirname +'/'+classes+'/'+filename)
            cv2.imwrite(datapath + '/' + foldername + '/' + 'train' + '/' + classes + '/' + filename, image)
          elif count < number * val_rate:
            image = cv2.imread(dirname + '/' +classes+'/' +filename)
            cv2.imwrite(datapath + '/' + foldername + '/' + 'valid' + '/' + classes + '/' + filename, image)
          else:
            image = cv2.imread(dirname + '/' + classes + '/' + filename)
            cv2.imwrite(datapath + '/' + foldername + '/' + 'test' + '/' + classes + '/' + filename, image)
        count += 1
def makedirs(dirname, classes):
  path = os.getcwd()
  if not os.path.isdir(path +'/' + dirname):
    os.makedirs(path + '/' + dirname)
  if not os.path.isdir(path +'/' + dirname + '/' + 'train'):
    os.makedirs(path + '/' + dirname + '/' + 'train')
  if not os.path.isdir(path +'/' + dirname + '/' + 'valid'):
    os.makedirs(path + '/' + dirname + '/' + 'valid')
  if not os.path.isdir(path +'/' + dirname + '/' + 'test'):
    os.makedirs(path + '/' + dirname + '/' + 'test')
  if not os.path.isdir(path +'/' + dirname + '/' + 'train' + '/' + classes):
    os.makedirs(path +'/' + dirname + '/' + 'train' + '/' + classes)
  if not os.path.isdir(path +'/' + dirname + '/' + 'valid' + '/' + classes):
    os.makedirs(path + '/' + dirname + '/' + 'valid' + '/' + classes)
  if not os.path.isdir(path +'/' + dirname + '/' + 'test' + '/' + classes):
    os.makedirs(path + '/' + dirname + '/' + 'test' + '/' + classes)



makedirs('data_natural_image', 'car')
makedirs('data_natural_image', 'airplane')
makedirs('data_natural_image', 'cat')
makedirs('data_natural_image', 'dog')
makedirs('data_natural_image', 'motorbike')
makedirs('data_natural_image', 'flower')
makedirs('data_natural_image', 'fruit')
makedirs('data_natural_image', 'person')

path = 'C:\\Users\\subak\\PycharmProjects\\project20210517\\data (2)\\8-multi-class_data\\natural_images'
train_val_spliter(path, 'data_natural_image', 'car')
train_val_spliter(path, 'data_natural_image', 'airplane')
train_val_spliter(path, 'data_natural_image', 'cat')
train_val_spliter(path, 'data_natural_image', 'dog')
train_val_spliter(path, 'data_natural_image', 'flower')
train_val_spliter(path, 'data_natural_image', 'motorbike')
train_val_spliter(path, 'data_natural_image', 'fruit')
train_val_spliter(path, 'data_natural_image', 'person')





