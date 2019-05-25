import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

def larger_model(num_classes=10):
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

number_of_classes = 10

(images_train, labels_train), (images_test, labels_test) = mnist.load_data()

images_train = images_train.reshape(images_train.shape[0], images_train.shape[1], images_train.shape[2], 1).astype('float32')
images_test  = images_test.reshape(images_test.shape[0], images_test.shape[1], images_test.shape[2], 1).astype('float32')
images_train /= 255
images_test /= 255

print(labels_train[1])
labels_train  = np_utils.to_categorical(labels_train, number_of_classes)
labels_test   = np_utils.to_categorical(labels_test , number_of_classes)
print(labels_train[1][0])
exit()
model = larger_model(number_of_classes)

model.fit(images_train, labels_train, validation_data=(images_test, labels_test), epochs=20, batch_size=200)

model.save('./models/mnistCNN_num.model')

metrics = model.evaluate(images_test, labels_test, verbose=0)
print("Metrics(Test loss & Test Accuracy): ")
print(metrics)
