from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import keras

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)

num_classes = 10  # Tổng số lớp của MNIST (các số từ 0-9)
n_hidden_1 = 256  # layer thứ nhất với 256 neurons
n_hidden_2 = 256  # layer thứ hai với 256 neurons
num_input = 784  # Số features đầu vào (tập MNIST với shape: 28*28)
learning_rate = 0.1
num_epoch = 10
batch_size = 128

def prf(test, pred):
    return precision_score(test, pred, average='macro'), recall_score(test, pred, average='macro'), f1_score(test, pred, average='macro')

def evaluate(layers):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    train_size = 8/10
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=11)
    size=7/8
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=size, random_state=11)

    x_train = x_train/255
    x_test = x_test/255
    x_val= x_val/255
    # one hot

    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
    y_val = keras.utils.np_utils.to_categorical(y_val, num_classes)



    # build model
    zzz = Sequential()
    zzz.add(Flatten(input_shape=(28, 28)))
    for i in range(layers):
        zzz.add(Dense(n_hidden_1, activation='relu'))  # hidden layer1
    # model.add(Dense(n_hidden_2, activation='relu'))  # hidden layer2
    zzz.add(Dense(num_classes, activation='softmax'))  # output layer

    # loss, optimizers
    zzz.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=SGD(lr=learning_rate),
                  metrics=['accuracy'])

    hys = zzz.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), epochs=num_epoch)
    plot_model(zzz, to_file='model1.png')
    score = zzz.evaluate(x_test, y_test)
    y_pred = np.argmax(zzz.predict(x_test, batch_size=batch_size), axis=1)
    y_pred = keras.utils.np_utils.to_categorical(y_pred, num_classes)
    v=prf(y_test, y_pred)
    training_loss = hys.history["loss"]
    test_loss = hys.history["val_loss"]
    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    # Visualize loss history
    plt.plot(epoch_count, training_loss, "r--")
    plt.plot(epoch_count, test_loss, "b-")
    plt.legend(["Training Loss", "Validate Loss"])
    plt.xlabel("Epoch")
    plt.show()
    return score[0], score[1], v[0], v[1], v[2]

