import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

genre_list_dic = {'LSTM'  : ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'],
                  'LSTM_extra' : ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'],
                  'LSTM1' : ['strong', 'mild'],
                  'LSTM1_extra' : ['strong', 'mild'],
                  'LSTM2a': ['Sub-strong1', 'Sub-strong2'],
                  'LSTM2a_extra': ['Sub-strong1', 'Sub-strong2'],
                  'LSTM2b': ['Sub-mild1', 'Sub-mild2'],
                  'LSTM2b_extra': ['Sub-mild1', 'Sub-mild2'],
                  'LSTM3a': ['hiphop', 'metal', 'rock'],
                  'LSTM3a_extra': ['hiphop', 'metal', 'rock'],
                  'LSTM3b': ['pop', 'reggae'],
                  'LSTM3b_extra': ['pop', 'reggae'],
                  'LSTM3c': ['country', 'disco'],
                  'LSTM3c_extra': ['country', 'disco'],
                  'LSTM3d': ['blues', 'classical', 'jazz'],
                  'LSTM3d_extra': ['blues', 'classical', 'jazz'],
                  'Group': ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock', 'electronic', 'world'],
                  'GroupRoot': ['group_a', 'group_b', 'group_c', 'group_d', 'group_e'],
                  'GroupA':['blues', 'jazz'],
                  'GroupB':['rock', 'metal'],
                  'GroupE':['pop', 'hiphop','reggae'],
                  'GroupC':['classical', 'country', 'world'],
                  'Test':['classical', 'country', 'world'],
                  'GroupD':['electronic', 'disco'],
                  'emotion':['happy', 'sad', 'relax', 'angry']}

binary_classification_list = ['LSTM1', 'LSTM2a', 'LSTM2b', 'LSTM3b', 'LSTM3c', 'GroupA', 'GroupB', 'GroupD']

def one_hot(Y_genre_strings, genre_list):
    y_one_hot = np.zeros((Y_genre_strings.shape[0], len(genre_list)))
    for i, genre_string in enumerate(Y_genre_strings):
        index = genre_list.index(genre_string)
        y_one_hot[i, index] = 1
    return y_one_hot

def decode(data, model):
    index = np.argmax(data)
    return genre_list_dic[model][index]

def load_data(model_name):
    genre_list = genre_list_dic[model_name]
    X_data = './MFCC_dataset/%s_x_data.npy' % model_name
    y_data = './MFCC_dataset/%s_y_data.npy' % model_name
    X_data = np.load(X_data)
    y_data = np.load(y_data)
    y_data = one_hot(y_data, genre_list)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30, shuffle=True, stratify=y_data)
    # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30, shuffle=True, random_state=1, stratify=y_data)
    return X_train, X_test, y_train, y_test
