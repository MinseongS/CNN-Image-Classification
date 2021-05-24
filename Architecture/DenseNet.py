from tensorflow.keras.models import Model
import keras.backend as K
from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, MaxPooling2D, AveragePooling2D, BatchNormalization, \
    ELU, Reshape, Concatenate, Activation, GlobalAveragePooling2D, Dense
from tensorflow.keras.regularizers import l2

def _bn_relu_conv(x, growth_rate, dropout_rate=None, weight_decay=1E-4):
    x = BatchNormalization(axis=-1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate*4, (1, 1), kernel_initializer='he_normal', padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def transition(x, nb_filter, dropout_rate=None, weight_decay=1E-4, pooling=True):
    x = BatchNormalization(axis=-1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    if pooling:
        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    return x

def denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    for i in range(nb_layers):
        merge_tensor = _bn_relu_conv(x, growth_rate, dropout_rate, weight_decay)
        x = Concatenate(axis=-1)([merge_tensor, x])
        nb_filter += growth_rate
    return x, nb_filter


def get_model(input_tensor, dropout_rate, weight_decay, nb_filter, growth_rate):

    # densnet121
    conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding="same", kernel_initializer='he_normal',kernel_regularizer=l2(0.0005), name='conv1')(input_tensor)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1)  # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    conv1 = ELU(name='elu1')(conv1)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name='pool1')(conv1)

    conv2, nb_filter = denseblock(pool1, 6, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    conv2_tr = transition(conv2, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay, pooling=True)

    conv3, nb_filter = denseblock(conv2_tr, 12, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    conv3_tr = transition(conv3, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay, pooling=True)

    conv4, nb_filter = denseblock(conv3_tr, 24, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    conv4_tr = transition(conv4, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay, pooling=False)

    conv5, nb_filter = denseblock(conv4_tr, 16, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    conv5_tr = transition(conv5, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay, pooling=False)

    x = GlobalAveragePooling2D(name='avg_pool')(conv5_tr)
    x = Dense(1000, activation='softmax', name='fc1000')(x)

    model = Model(input_tensor, x, name='densenet_final')

    return model





dropout_rate=0.3
weight_decay = 0.0001
nb_filter = 64
growth_rate = 32

x = Input(shape=(224, 224, 3))
model_init = get_model(input_tensor=x, dropout_rate=dropout_rate, weight_decay=weight_decay, nb_filter=nb_filter, growth_rate=growth_rate)
model_init.summary()
print("생성완료")

