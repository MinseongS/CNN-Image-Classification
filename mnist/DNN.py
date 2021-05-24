from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np

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

batch_size = 128
num_classes = 10
epochs = 12

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(np.reshape(X_train[i], [28, 28]), cmap='Greys')
#     plt.xlabel(y_train[i])
# plt.show()
L, W, H = X_train.shape
X_train = X_train.reshape(60000, W * H)
X_test = X_test.reshape(10000, W * H)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# print("y_train_ori:{}".format(y_train[:5]))
y_train = to_categorical(y_train, num_classes)
# print("y_train_after:{}".format(y_train[:5]))
y_test = to_categorical(y_test, num_classes)

input_tensor = Input(shape=(784,), name='input_tensor')
hidden_1 = Dense(units=256, activation='sigmoid', name='hidden_1')(input_tensor)
hidden_2 = Dense(units=256, activation='sigmoid', name='hidden_2')(hidden_1)
hidden_3 = Dense(units=256, activation='sigmoid', name='hidden_3')(hidden_2)
hidden_4 = Dense(units=256, activation='sigmoid', name='hidden_4')(hidden_3)
output_tensor = Dense(num_classes, activation='softmax', name='output_tensor')(hidden_4)
model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2)
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print("\nTest loss:", score[0])
print("Test acc:", score[1])

plt.subplot(1, 2, 1)
plot_loss(history)
plt.subplot(1, 2, 2)
plot_acc(history)
plt.show()
