from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
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
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
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

np.random.seed(1337)

batch_size = 128
nb_classes = 10
nb_epochs = 12

img_rows, img_cols = 28, 28
nb_filters = 32

(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)


# 연쇄형
# model = Sequential()
# model.add(Conv2D(nb_filters, kernel_size=(3, 3), padding='valid', input_shape=input_shape, name='Conv1'))
# model.add(Activation('relu', name='relu_1'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Conv2D(nb_filters, kernel_size=(3, 3), name='Conv2'))
# model.add(Activation('relu', name='relu_2'))
# model.add(MaxPooling2D(pool_size=(2, 2), name='pool_1'))
# model.add(Flatten())
# model.add(Dense(128, name='hidden_1'))
# model.add(Activation('relu', name='relu_3'))
# model.add(Dense(nb_classes, name='hidden_2'))
# model.add(Activation('softmax', name='output_tensor'))
# model.summary()

# 함수형
input_tensor = Input(shape=input_shape)
x = Conv2D(nb_filters, kernel_size=(3, 3), padding='valid', name='Conv1')(input_tensor)
x = BatchNormalization()(x)
x = Activation('relu', name='relu_1')(x)
x = Dropout(0.5)(x)
x = Conv2D(nb_filters, kernel_size=(3, 3), padding='valid', name='Conv2')(x)
x = Activation('relu', name='relu_2')(x)
x = MaxPooling2D(pool_size=(2, 2), name='pool_1')(x)
x = Flatten()(x)
x = Dense(units=128, name='hidden_1')(x)
x = Activation('relu', name='relu_3')(x)
x = Dense(units=nb_classes, name='hidden_2')(x)
output_tensor = Activation('softmax', name='output_tensor')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=nb_epochs,
                    verbose=1,
                    validation_split=0.2)
print('Test start')
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.subplot(1, 2, 1)
plot_loss(history)
plt.subplot(1, 2, 2)
plot_acc(history)
plt.show()

