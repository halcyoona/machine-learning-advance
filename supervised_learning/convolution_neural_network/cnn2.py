from __future__ import print_function
import numpy as np
np.random.seed(1337)  #for reproducibility

from keras.dataset import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolutional2D, MaxPooling
from keras.optimizer=rms 

model.fit(X_train, Y_train)import SGD, Adam, RMSprop
from Keras.utils import np_utils

batch_size = 128
nb_classes = 10 
nb_epoch = 20

#image dimension
img_rows, img_cols = 28, 28

#number of convolutional filters to use
nb_filters = 32

#size of pooling area for max pooling
nb_pool

#convolutional kernel size 
nb_conv = 3



#data shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


X_train = X_train.reshape(X_train,shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_train,shape[0], 1, img_rows, img_cols)

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

model.add(Convolutional2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1,img_rows,img_cols)))
model.add(Activation("relu"))
model.add(Convolutional2D(nb_filters, nb_conv, nb_conv))
model.add(Activation("relu"))
model.add(MaxPooling(pool_size=(nb_pool, nb_pool)))

model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Drop(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, show_accuracy, verbose=0)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])
