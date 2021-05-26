from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Input, Conv3D, Add, LeakyReLU,AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2


def residual_block(x, filters_in, filters_out, k_size):
    shortcut = x
    x = Conv2D(filters_in, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters_in, kernel_size=(k_size, k_size), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters_out, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)

    shortcut_channel = shortcut.shape.as_list()[-1]

    if shortcut_channel != filters_out:
        shortcut = Conv2D(filters_out, kernel_size=(1, 1), strides=(1, 1), padding="same")(shortcut)

    x = Add()([x, shortcut])
    return LeakyReLU()(x)

def define_model(image_height, image_width, image_channel, classes):
    input_tensor = Input(shape=(image_height, image_width, image_channel))
    x = Conv2D(64, kernel_size=(7, 7), strides=2, padding='same',
               kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = residual_block(x, 64, 256, 3)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = residual_block(x, 128, 512, 3)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = residual_block(x, 256, 1024, 3)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = residual_block(x, 512, 2048, 3)
    x = AveragePooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = define_model(227, 227, 3, 1000)
model.summary()
