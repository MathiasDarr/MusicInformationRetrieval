"""
This script trains the keras model
"""
# !/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from math import floor

import psutil
from keras.callbacks import Callback
from random import shuffle
import numpy as np
from librosa import time_to_frames
import boto3
import os
import time


def load_transform_and_annotation(path):
    path = '{}'.format(path)
    # annotation_label = np.load(path+'annotation.npy') if binary else np.load(path+'multivariable_annotation.npy')
    cqt = np.load('{}/cqt.npy'.format(path))
    annotation_matrix = np.load('{}/annotation_matrix.npy'.format(path))
    return cqt, annotation_matrix


# folders = ['2004', '2006', '2008', '2009', '2011','2013','2014','2015', '2017','2018']
# folders = ['2013']
# transform_directories = os.listdir('data/maestro/2013')
# cqt, annotation = load_transform_and_annotation('{}/{}'.format('data/maestro/2013',transform_directories[0]))


def maestroGenerator(batchsize, train=True):
    def init_file_queue():


        files = ['{}/{}'.format('data/maestro', file) for file in os.listdir('data/maestro')]
        files = files[:40]
        os.listdir('data/maestro')
        nfiles = len(files)
        nfiles75 = int(nfiles * .75)
        training_files = files[:nfiles75]
        test_files = files[nfiles75:]

        if train:
            training_files = list(training_files)
            return training_files
        else:
            test_files = list(test_files)
            return test_files


    def stitch(next_spec, next_annotation):
        '''
        This method will handle the case when the generator reaches the end of one spectrogram and stitch together
        the samples from the next.
            Calculate how many samples of the next spectogram I need to grab. Then set the current_spectogra_index to this value
            This method will be called when the spectogram gets pulled off the queue requiring the need to stitch together the spectograms
        '''

        n_samples = batchsize + currentIndex - x.shape[0]  # Number of samples in next spectogram
        prev_n_samples = batchsize - n_samples  # Number of samples in the previous spectogram

        spec1 = x[-prev_n_samples:]
        spec2 = next_spec[:n_samples]
        # print("The shapes of the spec {} and {}".format(spec1.shape, spec2.shape))
        batchx = np.concatenate((spec1, spec2), axis=0)

        annotation1 = y[-prev_n_samples:]
        annotation2 = next_annotation[:n_samples]
        batchy = np.concatenate((annotation1, annotation2), axis=0)

        return batchx, batchy, next_spec, next_annotation, n_samples

    def generate_windowed_samples(spec):
        '''
        This method creates the context window for a sample at time t, Wi-2, Wi-1, Wi, Wi+1,Wi+2
        '''
        windowed_samples = np.zeros((spec.shape[0], 5, spec.shape[1]))
        for i in range(spec.shape[0]):
            if i <= 1:
                windowed_samples[i] = np.zeros((5, spec.shape[1]))
            elif i >= spec.shape[0] - 2:
                windowed_samples[i] = np.zeros((5, spec.shape[1]))
            else:
                windowed_samples[i, 0] = spec[i - 2]
                windowed_samples[i, 1] = spec[i - i]
                windowed_samples[i, 2] = spec[i]
                windowed_samples[i, 3] = spec[i + 1]
                windowed_samples[i, 4] = spec[i + 2]
        return windowed_samples

    welford_mean = np.load('welford_mean.npy')
    welford_variance = np.load('welford_variance.npy')
    welford_standard_deviation = np.sqrt(welford_variance)

    fileQueue = init_file_queue()
    filedirectory = fileQueue.pop()
    x, y = load_transform_and_annotation(filedirectory)
    x = (x - welford_mean) / welford_standard_deviation
    x = generate_windowed_samples(x)

    currentIndex = 0

    while True:
        if currentIndex > x.shape[0] - batchsize:
            if len(fileQueue) == 0:
                fileQueue = init_file_queue()
            next_spec_id = fileQueue.pop()
            # print("Processing the next fiel with id {}".format(next_spec_id))
            # print("Length of the queue is {}".format(len(fileQueue)))
            nextSpec, annotation_matrix = load_transform_and_annotation(next_spec_id)
            nextSpec = generate_windowed_samples(nextSpec)

            nextSpec = (nextSpec - welford_mean) / welford_standard_deviation

            batchx, batchy, x, y, currentIndex = stitch(nextSpec,annotation_matrix)

            yield batchx.reshape((batchx.shape[0], batchx.shape[1], batchx.shape[2], 1)), batchy
        else:
            batchx = x[currentIndex:currentIndex + batchsize]
            batchy = y[currentIndex:(currentIndex + batchsize)]
            currentIndex = currentIndex + batchsize
            yield batchx.reshape((batchx.shape[0], batchx.shape[1], batchx.shape[2], 1)), batchy


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='normal', activation='relu', padding='same',
                     input_shape=(5, 84, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(84, kernel_initializer='normal', activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# download_guitarset_transforms()

batch_size = 32
model = build_model()
num_epochs = 10

model.fit_generator(generator=maestroGenerator(32),
                    epochs=num_epochs,
                    steps_per_epoch=floor(8382182 / batch_size),
                    verbose=1,
                    use_multiprocessing=False,
                    workers=16,
                    validation_data=maestroGenerator(32, False),
                    validation_steps=floor(888281 / batch_size),
                    # callbacks=[CustomCallback()],
                    max_queue_size=32)
