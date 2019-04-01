import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam
from utils import decode
from sklearn import preprocessing
import librosa
import librosa.display
import re
import matplotlib.pyplot as plt


def load_model(model_path, weights_path, binary_classification):
    "Load the trained LSTM model from directory for genre classification"
    with open(model_path, 'r') as model_file:
        trained_model = model_from_json(model_file.read())
    trained_model.load_weights(weights_path)
    if binary_classification:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    trained_model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])
    return trained_model

#link 7 LSTMs together hierarchically
def hierarchial_learning(data):
    LSTM1 = load_model("./LSTM1.json","./LSTM1.h5", True)
    LSTM2a = load_model("./LSTM2a.json","./LSTM2a.h5", True)
    LSTM2b = load_model("./LSTM2b.json","./LSTM2b.h5", True)
    LSTM3a = load_model("./LSTM3a.json","./LSTM3a.h5", False)
    LSTM3b = load_model("./LSTM3b.json","./LSTM3b.h5", True)
    LSTM3c = load_model("./LSTM3c.json","./LSTM3c.h5", True)
    LSTM3d = load_model("./LSTM3d.json","./LSTM3d.h5", False)
    result1 = LSTM1.predict(data)
    result1 = decode(result1, 'LSTM1')
    if result1 == 'strong':
        result2 = LSTM2a.predict(data)
        result2 = decode(result2, 'LSTM2a')
        if result2 == 'Sub-strong1':
            result3 = LSTM3a.predict(data)
            return decode(result3, 'LSTM3a')
        else:
            result3 = LSTM3b.predict(data)
            return decode(result3, 'LSTM3b')
    else:
        result2 = LSTM2b.predict(data)
        result2 = decode(result2, 'LSTM2b')
        if result2 == 'Sub-mild1':
            result3 = LSTM3c.predict(data)
            return decode(result3, 'LSTM3c')
        else:
            result3 = LSTM3d.predict(data)
            return decode(result3, 'LSTM3d')


def extract_audio_features(filepath):
    n_frame = 1290
    n_mfcc = 14
    data = np.zeros((1, n_mfcc, n_frame), dtype=np.float64)
    target = []
    y, sr = librosa.load(filepath)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    filename = re.split('[/]', filepath)[-1]
    genre = re.split('[ .]', filename)[0]
    print("filename %s, genre %s" % (filename, genre))
    target.append(genre)
    print(np.shape(mfcc))
    data[0] = mfcc[:, :n_frame]
    print("Extracted features audio track %s" % filename)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig('%s.png'%filename)
    return data

def main():
    data = extract_audio_features("./Test/blues.00000.au")
    result = hierarchial_learning(data)
    print(result)

if __name__ == '__main__':
    main()
