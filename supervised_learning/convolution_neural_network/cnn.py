from __future__ import print_function
import numpy as np
np.random.seed(1337)  #for reproducibility

from keras.dataset import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from Keras.utils import np_utils

batch_size = 128
nb_classes = 10 
nb_epoch = 20


#data shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#reshaping just for the model to run
#28x28 = 784
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalizing from 0 to 1
X_train /= 255
X_test /= 255
print(X_train.shape[0], "Train Samples")
print(X_test.shape[0], "Test Samples")


#convert class vectors into binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

model = Sequential()

model.add(Dense(512, input_shape=(784, )))
model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(10))

model.add(Activation('softmax'))


rms = RMSprop()

model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, show_accuracy, verbose=0)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])



