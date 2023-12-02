import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

import bindata

def batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    return num_batches_per_epoch, bindata.fit_generator()


def main(mode):
    max_features = 20000
    maxlen = 80
    batch_size = 32

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if mode == 'fit':
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))
    else:
        train_steps, train_batches = batch_iter(x_train, y_train, batch_size)
        valid_steps, valid_batches = batch_iter(x_test, y_test, batch_size)
        model.fit_generator(train_batches, train_steps, epochs=1, validation_data=valid_batches, validation_steps=valid_steps)


if __name__ == '__main__':
    import sys
    mode = 'fit'
    main(mode)
