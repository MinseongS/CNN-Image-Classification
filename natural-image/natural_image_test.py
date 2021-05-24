from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def search(dirname):
    img_list = []
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg':
                img_list.append(path+'/'+filename)
    return img_list

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(200, 200))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = np.expand_dims(img, axis=0)
    # img scaling
    img /= 255

    return img

img_list_1 = search('data_natural_image/test/temp')

model = load_model('./natural_image_model.h5')

for i in img_list_1:
    img = load_image(i)
    # predict the class
    result = model.predict(img)
    preds_value = np.argmax(result, axis=-1)[0]

    print(preds_value)

