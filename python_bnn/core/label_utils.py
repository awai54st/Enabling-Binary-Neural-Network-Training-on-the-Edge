import numpy as np


def label_binerizer(labels, n_class=None):
    if n_class is None:
        binarized_label = np.zeros((len(labels), len(np.unique(train_labels))))
    else:
        binarized_label = np.zeros((len(labels), n_class))
    binarized_label[range(len(labels)), labels] = 1
    return binarized_label