import librosa
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def get_waveform(filepath):
    y, sr = librosa.load(filepath)
    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.waveplot(y=y, sr=sr)
    filename = re.split('[/]', filepath)[-1]
    plt.title('Waveform - %s' % filename)
    plt.savefig('./Audio Visualization/Waveform/%s.png'% filename)
    plt.clf()

def get_chromagram(filepath):
    y, sr = librosa.load(filepath)
    librosa.feature.chroma_stft(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    filename = re.split('[/]', filepath)[-1]
    plt.title('Chromagram - &s' % filename)
    plt.savefig('./Audio Visualization/Chromagram/%s.png'% filename)
    plt.clf()

def get_mfcc(filepath):
    y, sr = librosa.load(filepath)
    librosa.feature.mfcc(y=y, sr=sr, n_mfcc=14)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.savefig('./Audio Visualization/MFCC/%s.png'% filename)
    plt.clf()
