import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import math
import re
import os
import sys

n_frame = 128
n_mfcc = 14

def path_to_audiofiles(dataset):
    path = "./%s" % dataset
    list_of_audiofiles = []
    for file in os.listdir(path):
        if file.endswith(".au") or file.endswith(".wav"):
            directory = "%s/%s" % (path, file)
            list_of_audiofiles.append(directory)
    return list_of_audiofiles

def extract_audio_features(list_of_audiofiles, dataset, extra):
    X_data = './MFCC_dataset/%s_x_data' % dataset
    y_data = './MFCC_dataset/%s_y_data' % dataset
    #calculate the total number of features
    n_feature = n_mfcc + int(extra)*12
    data = np.zeros((len(list_of_audiofiles), n_frame, n_feature), dtype=np.float64)
    target = []
    for i, filepath in enumerate(list_of_audiofiles):
        y, sr = librosa.load(filepath)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        data[i,:n_frame,:n_mfcc] = mfcc.T[:n_frame, :]
        #if add_spectral_centroid:
        #    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        #    data[i,:n_frame,n_mfcc:n_mfcc+1] = spectral_centroid.T[:n_frame,:]
        #if add_chroma:
        #    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        #    data[i,:n_frame,n_mfcc+1:] = chroma.T[:n_frame,:]

        filename = re.split('[/]', filepath)[-1]
        genre = re.split('[ .]', filename)[0]
        print("filename %s, genre %s" % (filename, genre))
        target.append(genre)
        print(np.shape(mfcc))
        print("Extracted features audio track %s" % filename)

    target = np.expand_dims(np.asarray(target), axis=1)

    if add_spectral_centroid:
        X_data += '_sc'
        y_data += '_sc'
    if add_chroma:
        X_data += '_cr'
        y_data += '_cr'        
    X_data += '.npy'
    y_data += '.npy'
    with open(X_data, 'wb+') as f:
        np.save(f, data)
    with open(y_data, 'wb+') as f:
        np.save(f, target)

def main():
    dataset = sys.argv[1]
    add_spectral_centroid = (sys.argv[2] == 'True')
    add_chroma = (sys.argv[3] == 'True')
    p = path_to_audiofiles(dataset)
    extract_audio_features(p, dataset, add_spectral_centroid, add_chroma)

if __name__ == '__main__':
    main()
