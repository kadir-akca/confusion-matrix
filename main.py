import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# Stuttgart Uni.

def confusion_matrix(y_true, y_pred, normalize=None):
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")

    # Find n_labels by determining the max value (if min is 0)
    n_labels = np.maximum(np.max(y_true), np.max(y_pred)) + 1
    print(n_labels)

    conf_matr = np.zeros((n_labels, n_labels))

    for i, j in zip(y_pred, y_true):
        conf_matr[i, j] += 1
        print(conf_matr[i, j])

    if normalize == 'true':
        conf_matr = conf_matr / conf_matr.sum(axis=0, keepdims=True)
    elif normalize == 'pred':
        conf_matr = conf_matr / conf_matr.sum(axis=1, keepdims=True)
    elif normalize == 'all':
        conf_matr = conf_matr / conf_matr.sum()

    return conf_matr


def precision(y_true, y_pred):
    pass


def recall(y_true, y_pred):
    pass


def false_alarm_rate(y_true, y_pred):
    pass


def cm():
    confusion_matrix_l = metrics.confusion_matrix(act, pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_l, display_labels=[False, True, 'a'])
    cm_display.plot()
    plt.show()


actual = np.random.binomial(1, 0.9, size=1000)
predicted = np.random.binomial(1, 0.9, size=1000)

act = np.random.randint(3, size=100)
pred = np.random.randint(3, size=100)

confusion_matrix(act, pred)
