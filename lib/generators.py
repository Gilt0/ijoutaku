import os
import random
import time

import keras

import PIL as pil
import numpy as np


class SequenceDataGenerator(keras.utils.Sequence):
    
    _IMAGE_WIDTH = 128
    _IMAGE_HEIGHT = 88
    _TIF_EXTENSION = '.tif'
    _STRIDE_KERNEL = 5
    _BATCH_SIZE = 3
    _SEQUENCE_SIZE = 200

    def __init__(self, data_path):
        self._data_path = data_path
        self._sequences = list()
        self._batches = list()
        self._len = 0
        self.__load__()

    def __make_batches__(self, ):
        strided = list()
        for sequence in self._sequences:
            if not os.path.isdir(sequence): continue
            image_files = [ f'{sequence}{image_file}' for image_file in sorted(os.listdir(sequence)) ]
            if not image_files: continue
            for stride in range(self._STRIDE_KERNEL):
                strided.append(image_files[stride::self._STRIDE_KERNEL])

        self._batches = [ strided[b:b+self._BATCH_SIZE] for b in range(0, len(strided), self._BATCH_SIZE) ]
        self._len = len(self._batches)

    def __load__(self, ):
        self._sequences = sorted([ f'{self._data_path}{data_folder}/' for data_folder in os.listdir(self._data_path) ])
        self.__make_batches__()

    def __len__(self, ):
        return self._len

    def __getitem__(self, index):
        X = np.zeros((self._BATCH_SIZE, int(self._SEQUENCE_SIZE/self._STRIDE_KERNEL), self._IMAGE_HEIGHT, self._IMAGE_WIDTH, 1), dtype=np.float16)
        batch = self._batches[index]
        for b, image_paths in enumerate(batch):
            for t, image_path in enumerate(image_paths):
                image = np.array(pil.Image.open(image_path).resize((self._IMAGE_WIDTH, self._IMAGE_HEIGHT)), dtype=np.float16)/256
                X[b, t, :, :, 0] = image
        return X, X

    def on_epoch_end(self, ):
        random.shuffle(self._sequences)
        self.__make_batches__()


class ForwardDataGenerator(keras.utils.Sequence):
    
    _IMAGE_WIDTH = 128
    _IMAGE_HEIGHT = 88
    _TIF_EXTENSION = '.tif'
    _LOOKBACK = 5
    _BATCH_SIZE = 3
    _SEQUENCE_SIZE = 200

    def __init__(self, data_path, shuffle_at_start=False):
        self._data_path = data_path
        self._shuffle_at_start = shuffle_at_start
        self._sequences = list()
        self._batches = list()
        self._len = 0
        self.__load__()

    def __make_batches__(self, ):
        self._batches = [ self._sequences[i:i+self._BATCH_SIZE] for i in range(0, len(self._sequences), self._BATCH_SIZE) ]
        self._len = len(self._batches)

    def __load__(self, ):
        tmp_sequences = sorted([ f'{self._data_path}{data_folder}/' for data_folder in os.listdir(self._data_path) ])
        for sequence in tmp_sequences:
            if not os.path.isdir(sequence): continue
            image_files = sorted([ f'{sequence}{image_file}' for image_file in os.listdir(sequence) if image_file.find('.tif') > -1 ])
            if not image_files: continue
            for n in range(len(image_files) - self._LOOKBACK):
                predictors = image_files[n:n+self._LOOKBACK]
                predicted = image_files[n+self._LOOKBACK]
                self._sequences.append((predictors, predicted))
        if self._shuffle_at_start: random.shuffle(self._sequences)
        self.__make_batches__()

    def __len__(self, ):
        return self._len

    def __getitem__(self, index):
        X = np.zeros((self._BATCH_SIZE, self._LOOKBACK, self._IMAGE_HEIGHT, self._IMAGE_WIDTH, 1), dtype=np.float16)
        y = np.zeros((self._BATCH_SIZE, self._IMAGE_HEIGHT, self._IMAGE_WIDTH, 1), dtype=np.float16)
        batch = self._batches[index]
        for b, (predictors, predicted) in enumerate(batch):
            for l, predictor in enumerate(predictors):
                X[b, l, :, :, 0] = np.array(pil.Image.open(predictor).resize((self._IMAGE_WIDTH, self._IMAGE_HEIGHT)), dtype=np.float16)/256
            y[b, :, :, 0] = np.array(pil.Image.open(predicted).resize((self._IMAGE_WIDTH, self._IMAGE_HEIGHT)), dtype=np.float16)/256
        return X, y

    def on_epoch_end(self, ):
        random.shuffle(self._sequences)
        self.__make_batches__()