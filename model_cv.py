import os
import sys
import numpy as np
import json
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
import logging
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from utils import load_data, decode, binary_classification_list

# switch off tensorflow verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def get_configuration(filename):
   with open(filename) as f_in:
       return(json.load(f_in))

def create_binary_model(input_shape, num_output, batch_size, epochs, learning_rate, dropout):
    model = Sequential()
    model.add(LSTM(units=128, dropout=dropout, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=32, dropout=dropout, return_sequences=True))
    model.add(LSTM(units=num_output, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def create_nonbinary_model(input_shape, num_output, batch_size, epochs, learning_rate, dropout):
    model = Sequential()
    model.add(LSTM(units=128, dropout=dropout, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=32, dropout=dropout, return_sequences=True))
    model.add(LSTM(units=num_output, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def run(model_name, cv):
    logging.basicConfig(filename='./Log/%s-CV' % model_name, level=logging.INFO)
    X_train, _, y_train, _= load_data(model_name)
    input_shape = (np.shape(X_train)[1], np.shape(X_train)[2])
    model = KerasClassifier(build_fn=create_nonbinary_model, verbose=1)
    if model_name in binary_classification_list:
        model = KerasClassifier(build_fn=create_binary_model, verbose=1)

    param_grid = dict(get_configuration('./CV Configurations/%s.json' % model_name))
    param_grid['input_shape'] = [input_shape]
    param_grid['num_output'] = [y_train.shape[1]]
    grid = GridSearchCV(cv=cv, estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    logging.info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        logging.info("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == '__main__':
    model_name = sys.argv[1]
    cv = int(sys.argv[2])
    run(model_name, cv)
