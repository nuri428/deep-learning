from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
import pandas
import sys
import os
import cv2, numpy as np

nbClasses = 4
input_width = 224
input_height = 224

def VGG_16(nbOutput=1000):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nbOutput, activation='softmax'))
    return model

def load_data (data_path):
    columns = ['data','label']
    df = pandas.DataFrame(columns=columns)
    for root, subFolder, files in os.walk(data_path):
        for item in files:
            if item.endswith(".jpg") :
                fileNamePath = str(os.path.join(root,item))
                im = cv2.imread(fileNamePath)
                im = cv2.resize(im, (input_width,input_height))
                imgArray = np.asarray(im)
                classStr = int(''.join(fileNamePath.split("/")[-2:-1]))
                df.loc[len(df)] = [imgArray, classStr]

                if len(df) % 10000 == 0 :
                    print '%d load '%len(df)
    print 'load %d data '%len(df)
    data = np.array(df['data'].tolist())
    label = np.array(df['label'].tolist())
    data = np.transpose(data, (0,3,1,2))
    return data, label

if __name__ == "__main__":
    if len(sys.argv) == 1 :
        exit(-1)
    XTrain, YTrain = load_data(sys.argv[1])
    YTrain = np_utils.to_categorical(YTrain, nbClasses)

    model = VGG_16(nbClasses)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    print "now complile model"

    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(XTrain, YTrain, nb_epoch=10, batch_size=32)

    print "compile complete"
    model.save_weights('vgg-16.h5')
