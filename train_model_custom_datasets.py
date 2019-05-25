import os
import cv2
import numpy as np
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

def load_data(path):
    folders = os.listdir(path);print(folders)
    Len = [] #print(type(Len))
    images = []
    for i in range(0,len(folders)):
        folder = folders[i]#;print(folder)
        image_FileNames = os.listdir(path+'/' + folder+'/') #;print(path+'/' + folder+'/')
        #print(image_FileNames)
        data = []
        Len.append(len(image_FileNames))#;print(len(image_FileNames))
        for fileName in image_FileNames:
            image = cv2.imread(path +'/' + folder +'/' + fileName ,0);print(path +'/' + folder +'/' + fileName)
            data.append(image)
        images.extend(data)

    gen = []
    for i in range(0,len(Len)):
        gen.append(i)

    #print(gen)
    #print(Len)

    #print(type(gen))
    #print(type(Len))

    gen = tuple(gen)#;print(type(gen))
    Len = tuple(Len)#;print(type(Len))


    label = np.repeat(gen,Len)#;print(type(images));exit()
    images = np.array(images)

    return images, label, len(folders)

images_train , labels_train ,class_train = load_data('C:/Users/hp/Desktop/text_detection/datasets/datasets_train/character')
images_test ,  labels_test  ,class_test  = load_data('C:/Users/hp/Desktop/text_detection/datasets/datasets_test/character')

if class_train != class_test:
    exit()

images_train = images_train.reshape(images_train.shape[0], images_train.shape[1], images_train.shape[2], 1).astype('float32')
images_test  = images_test.reshape(images_test.shape[0], images_test.shape[1], images_test.shape[2], 1).astype('float32')

images_train /= 255
images_test /= 255

#print(labels_train[200])
labels_train  = np_utils.to_categorical(labels_train, class_train)
labels_test   = np_utils.to_categorical(labels_test , class_test)
#print(labels_train[200])

model = larger_model(class_train)

model.fit(images_train, labels_train, validation_data=(images_test, labels_test), epochs=20, batch_size=200)

metrics = model.evaluate(images_test, labels_test, verbose=0)
print("Metrics(Test loss & Test Accuracy): ")
print(metrics)

model.save('./models/mnistCNN_cha.model')