import os
import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
import logging
import matplotlib
import matplotlib.pyplot as plt
from audio_feature_extractor import extract_audio_features
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score, roc_curve, auc
import sys
from utils import load_data, decode, genre_list_dic, binary_classification_list
from plotter import confusion_matrix_plotter, accuracy_trace_plotter, loss_trace_plotter

# switch off tensorflows verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

def build_model(model_name, batch_size=35, n_epochs=20, learning_rate=0.001, dropout=0.1, num_test=10):
    model = None

    logging.basicConfig(filename='./Log/%s-testing-results' % model_name, filemode = 'a', level=logging.INFO)
    logging.info("batch_size:  %d" % batch_size)
    logging.info("n_epochs:  %d" % n_epochs)
    logging.info("learning_rate:  %.4f" % learning_rate)
    logging.info("dropout:  %.2f" % dropout)

    accuracy_scores = []
    loss_scores = []
    confusion_matrices = []
    history_val_acc = np.zeros(n_epochs)
    history_acc = np.zeros(n_epochs)
    history_val_loss = np.zeros(n_epochs)
    history_loss = np.zeros(n_epochs)

    for i in range(num_test):
        X_train, X_test, y_train, y_test = load_data(model_name)
        if model_name in binary_classification_list:
            #if its binary classification
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            activation = 'softmax'
            loss = 'categorical_crossentropy'

        input_shape = (np.shape(X_train)[1], np.shape(X_train)[2])
        print(input_shape)
        print('Build %s model ...' % model_name)
        model = Sequential()
        model.add(LSTM(units=128, dropout=dropout, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=32, dropout=dropout, return_sequences=True))
        model.add(LSTM(units=y_train.shape[1], activation=activation))
        #compile the model
        model.compile(loss=loss, optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        plot_model(model, to_file='%s.png' % model_name, show_shapes=True, show_layer_names=True)
        model.summary()

        #train and validate the model
        result = model.fit(X_train, y_train, validation_split=0.33, epochs=n_epochs, batch_size=batch_size)

        history_acc += np.array(result.history['acc'])
        history_val_acc += np.array(result.history['val_acc'])
        history_loss += np.array(result.history['loss'])
        history_val_loss += np.array(result.history['val_loss'])

        #test the model
        loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
        y_pred = model.predict(X_test)
        y_pred_decoded = []
        y_test_decoded = []
        for yp in y_pred:
            y_pred_decoded.append(decode(yp, model_name))
        for yt in y_test:
            y_test_decoded.append(decode(yt, model_name))

        print("Test %d Accuracy:  %.4f" % (i, accuracy))
        logging.info("Test %d Accuracy:  %.4f" % (i, accuracy))
        accuracy_scores.append(accuracy)
        print("Test %d Loss:  %.4f" % (i, loss))
        logging.info("Test %d Loss:  %.4f" % (i, loss))
        loss_scores.append(loss)
        cm = np.matrix(confusion_matrix(y_test_decoded, y_pred_decoded))
        confusion_matrices.append(cm)

    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    average_loss = sum(loss_scores) / len(loss_scores)

    print("Average Testing Accuracy:  %.4f" % average_accuracy)
    logging.info("Average Testing Accuracy:  %.4f" % average_accuracy)
    print("Average Testing Loss:  %.4f" % average_loss)
    logging.info("Average Testing Loss:  %.4f" % average_loss)

    #accuracy & loss
    accuracy_trace_plotter(model_name, history_acc, history_val_acc)
    loss_trace_plotter(model_name, history_loss, history_val_loss)

    #confusion matrix
    final_cm = np.zeros(confusion_matrices[0].shape)
    for cm in confusion_matrices:
        final_cm += cm
    confusion_matrix_plotter(final_cm, model_name)
    return model


def save_model(model):
    model_json = model.to_json()
    with open("./Trained Models/%s.json" % model_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights("./Trained Models/%s.h5" % model_name)
    print("Saved model to disk")


if __name__ == '__main__':
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    n_epochs = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    dropout = float(sys.argv[5])
    num_test = int(sys.argv[6])
    model = build_model(model_name, batch_size, n_epochs, learning_rate, dropout, num_test)
    save_model(model)
