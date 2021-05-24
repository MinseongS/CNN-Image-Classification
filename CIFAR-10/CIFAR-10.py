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
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'], loc=0)

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'], loc=0)

save_dir = os.path.join(os.getcwd(), 'saved_model')
model_name = 'keras_cifar10_aug_trained_model5.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = 10
batch_size = 128
epochs = 12

# classes = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
#
# actual_single = classes[y_train]
# plt.imshow(x_train[1000], interpolation='bicubic')
# tmp = "Label:" + str(actual_single[1000])
# plt.title(tmp, fontsize=30)
# plt.tight_layout()
# plt.show()

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

input_tensor = Input(shape=(32, 32, 3))
x = Conv2D(64, kernel_size=(5, 5), kernel_initializer='he_normal', padding='same')(input_tensor)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, kernel_size=(5, 5), kernel_initializer='he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(units=512)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(units=num_classes)(x)
output_tensor = Activation('softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


datagen = ImageDataGenerator(
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

#MODEL TRAIN(aug apply)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                              steps_per_epoch=x_train.shape[0]/16, epochs=epochs,
                              validation_data=(x_test, y_test), workers=4)

model_path = os.path.join(save_dir, model_name)
model.save(model_path)

#history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)

score = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest loss:", score[0])
print("Test acc:", score[1])

plt.subplot(1, 2, 1)
plot_loss(history)
plt.subplot(1, 2, 2)
plot_acc(history)
plt.show()
