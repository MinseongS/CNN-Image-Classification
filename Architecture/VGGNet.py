from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Input, Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
def define_model(image_height, image_width, image_channel, weight_decay, classes):
    input_tensor = Input(shape=(image_height, image_width, image_channel))
    x = Conv2D(64, kernel_size=(3, 3), padding='same',
               kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(input_tensor)
    x = Conv2D(64, kernel_size=(3, 3), padding='same',
               kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same',
               kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same',
               kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same',
               kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same',
               kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same',
               kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same',
               kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same',
               kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same',
               kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(units=4096, activation='relu')(x)
    x = Dense(units=4096, activation='relu')(x)
    output_tensor = Dense(classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = define_model(224, 224, 3, 0.01, 1000)
model.summary()