import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.utils.multiclass import unique_labels
from utils import genre_list_dic
def confusion_matrix_plotter(cm, model_name):

    classes = genre_list_dic[model_name]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Normalized Confusion Matrix - %s' % model_name,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    np.set_printoptions(precision=2)
    plt.savefig('./Confusion Matrix/%s.png'% model_name)
    plt.clf()

def accuracy_trace_plotter(model_name, acc, val_acc):
    # summarize history for accuracy
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('train & validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./Model Accuracy & Loss Diagram/%s-Accuracy'% (model_name))
    plt.clf()

def loss_trace_plotter(model_name, loss, val_loss):
    #summarize history for loss
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('train & validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./Model Accuracy & Loss Diagram/%s-Loss'% (model_name))
