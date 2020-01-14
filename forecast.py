from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from numpy import split
import pandas as pd
import numpy as np
from numpy import hstack
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
# demonstrate data normalization with sklearn
from sklearn.preprocessing import StandardScaler


# np.set_printoptions(threshold=np.nan)
# import os
# import tensorflow as tf
# import random as rn

# os.environ['PYTHONHASHSEED'] = '0'
# # Setting the seed for numpy-generated random numbers
# np.random.seed(37)
# # Setting the seed for python random numbers
# rn.seed(1254)
# # Setting the graph-level random seed.
# tf.set_random_seed(89)
# from keras import backend as K
# session_conf = tf.ConfigProto(
#       intra_op_parallelism_threads=1,
#       inter_op_parallelism_threads=1)
# #Force Tensorflow to use a single thread
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)


def split_train_test(dataset):
    train, test = dataset[:662], dataset[662:-3]
    print('length of training set', len(train))
    print('length of tests set', len(test))
    # print(train)
    # print(test)
    train = array(split(train, len(train)))
    test = array(split(test, len(test)))
    return train, test


def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        scores.append(rmse)
        # calculate overall RMSE
        s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
            score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=3):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0

    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end < len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])

        in_start += 1
    return array(X), array(y)


from keras import backend


def rmse(y_true, y_pred):
  return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# train the model
def build_model(train, n_input):
    train_x, train_y = to_supervised(train, n_input)
    test_x, test_y = to_supervised(test, n_input)
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)
    verbose, epochs, batch_size = 2, 20, 15

    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))
    #     train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    # define model
    model = Sequential()
    model.add(LSTM(1000, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(1000, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1000, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    # model.compile(optimizer='adam', loss='mse')
      model.compile(optimizer='adam', loss='mse',metrics=[rmse])
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        validation_data=(test_x, test_y))

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.title('loss', y=0, loc='center')
    pyplot.legend()
    pyplot.show()

      # plot rmse
    pyplot.plot(history.history['rmse'], label='train')
    pyplot.plot(history.history['val_rmse'], label='test')
    pyplot.title('rmse', y=0, loc='center')
    pyplot.legend()
    pyplot.show()
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# evaluate a single model
def evaluate_model(train, test, n_input):
    # fit model
    model = build_model(train, n_input)

    history = [x for x in train]

    predictions = list()
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        history.append(test[i, :])
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return model, score, scores, predictions


print('len of dataset', len(df_disc_temp))
df_disc_temp = df_disc_temp.fillna(method='ffill')
print(df_disc_temp.isnull().values.any())
print(df_disc_temp[662:-3].index.values)

# split into train and test
n_input = 7
train, test = split_train_test(df_disc_temp.values)

# # validate train data
print(train.shape)
# # print(train)
# print(train[0, 0, 0], train[-1, -1, 0])
# # validate test
print(test.shape)
# print(test[0, 0, 0], test[-1, -1, 0])


# evaluate model and get scores

model, score, scores, predictions = evaluate_model(train, test, n_input)
# # predictions=evaluate_model(train, test, n_input)
# print('prediction shape',predictions.shape)
print(predictions)
# np.savetxt("predicted.csv", predictions.reshape(-1,1), delimiter=",")


# # # # summarize scores
summarize_scores('lstm', score, scores)

# import pickle
# pkl_filename = "model.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(model, file)

