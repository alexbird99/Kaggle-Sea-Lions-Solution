"""
Solve sea lions counting problem as regression problem on whole image
"""
# import sys
import os
import keras
import numpy as np
import pandas as pd
import cv2
# import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
# from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
# from keras.layers import UpSampling2D
# from keras.layers import Input, concatenate
from keras.models import Model
# from keras import backend as K
from keras.layers import Activation
# from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.optimizers import SGD


N_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 100
IMAGE_SIZE = 512
MODEL_NAME = 'vgg_16_pre_trained'


def read_ignore_list():
    """
    test
    """
    df_ignore = pd.read_csv('kaggle_data/mismatched_train_images.txt')
    ignore_list = df_ignore['train_id'].tolist()
    return ignore_list


def load_data(dir_path):
    '''
    Just remove images from mismatched_train_images.txt
    '''
    train_data = pd.read_csv('kaggle_data/train.csv')

    ignore_list = read_ignore_list()
    n_train_images = 948

    image_list = []
    y_list = []
    for i in range(0, n_train_images):
        if i not in ignore_list:
            image_path = os.path.join(dir_path, str(i)+'.png')
            print image_path
            img = cv2.imread(image_path)
            print 'img.shape', img.shape
            image_list.append(img)

            row = train_data.ix[i]
            y_row = np.zeros((5))
            y_row[0] = row['adult_males']
            y_row[1] = row['subadult_males']
            y_row[2] = row['adult_females']
            y_row[3] = row['juveniles']
            y_row[4] = row['pups']
            y_list.append(y_row)

    x_train = np.asarray(image_list)
    y_train = np.asarray(y_list)

    print 'x_train.shape', x_train.shape
    print 'y_train.shape', y_train.shape

    return x_train, y_train


def get_model():
    """
    get model
    """
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(N_CLASSES, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())

    print model.summary()
    # sys.exit(0)

    model.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=keras.optimizers.Adadelta()
        )

    return model

def get_model_vgg_16():
    """
    vgg_16
    """
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, activation='softmax'))

    print model.summary

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def get_model_vgg_16_pre_trained():
    """
    get_model_vgg_16_pre_trained
    """
    vgg16= keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x= Conv2D(N_CLASSES, (1, 1), activation='relu')(vgg16.output)
    x= GlobalAveragePooling2D()(x)

    model = Model(vgg16.input, x)

    print model.summary()

    model.compile(loss=keras.losses.mean_squared_error,
            optimizer= keras.optimizers.Adadelta())

    return model



def train():
    """
    train
    """
    # model = get_model()
    # model = get_model_vgg_16()
    model = get_model_vgg_16_pre_trained()

    x_train, y_train = load_data('kaggle_data/train_images_512x512')
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(x_train) / BATCH_SIZE,
        epochs=EPOCHS
        )

    model.save(MODEL_NAME+'_model.h5')


def create_submission():
    """
    create submission
    """
    model = load_model(MODEL_NAME+'_model.h5')

    n_test_images = 18636
    pred_arr = np.zeros((n_test_images, N_CLASSES), np.int32)
    for k in range(0, n_test_images):
        image_path = 'kaggle_data/test_images_512x512/'+str(k)+'.png'
        print image_path

        img = cv2.imread(image_path)
        img = img[None, ...]
        pred = model.predict(img)
        pred = pred.astype(int)

        pred_arr[k, :] = pred

    print 'pred_arr.shape', pred_arr.shape
    pred_arr = pred_arr.clip(min=0)
    df_submission = pd.DataFrame()
    df_submission['test_id'] = range(0, n_test_images)
    df_submission['adult_males'] = pred_arr[:, 0]
    df_submission['subadult_males'] = pred_arr[:, 1]
    df_submission['adult_females'] = pred_arr[:, 2]
    df_submission['juveniles'] = pred_arr[:, 3]
    df_submission['pups'] = pred_arr[:, 4]
    df_submission.to_csv(MODEL_NAME+'_submission.csv', index=False)


train()
create_submission()
