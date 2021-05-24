from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'], loc=0)

# accuracy 그래프 그리는 함수
def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'], loc=0)

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', input_shape=(300, 300, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = define_model()

from keras.preprocessing.image import ImageDataGenerator

# create data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                   width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_it = train_datagen.flow_from_directory('data_natural_image/train/',
                                       class_mode='categorical', batch_size=64, target_size=(300, 300))
val_it = test_datagen.flow_from_directory('data_natural_image/valid/',
                                     class_mode='categorical', batch_size=64, target_size=(300, 300))
test_it = test_datagen.flow_from_directory('data_natural_image/test/',
                                     class_mode='categorical', batch_size=64, target_size=(300, 300))

# fit model
history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                              validation_data=val_it, validation_steps=len(val_it), epochs=20, verbose=1)

# evaluate model
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
print('>accuracy : %.3f' % (acc * 100.0))

# save model
model.save('natural_image_model2.h5')

plt.subplot(1, 2, 1)
plot_loss(history)
plt.subplot(1, 2, 2)
plot_acc(history)
plt.show()

