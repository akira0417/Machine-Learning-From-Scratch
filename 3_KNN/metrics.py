import numpy as np


def confusion_matrix(y_true, y_pred):
    """
    回傳：
            Pred 0   Pred 1
    True 0    TN       FP
    True 1    FN       TP
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([
        [tn, fp],
        [fn, tp]
    ])

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    tp = cm[1,1]
    fp = cm[0,1]

    if tp + fp == 0:
        return 0

    return tp / (tp + fp)

def recall_score(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    tp = cm[1,1]
    fn = cm[1,0]

    if tp + fn == 0:
        return 0

    return tp / (tp + fn)

def f1_score(y_true, y_pred):

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)