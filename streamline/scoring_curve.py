import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc, precision_recall_curve, average_precision_score


def score_roc_curve(y, probas_, classes):
    """
    Compute the micro averaged score for the ROCAUC curves.
    """
    # Convert y to binarized array for micro and macro scores
    y = label_binarize(y, classes=classes)
    if len(classes) == 2:
        y = np.hstack((1 - y, y))

    # Compute micro-average
    fpr, tpr, _ = roc_curve(y.ravel(), probas_.ravel())
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def score_precision_recall(y, probas_, classes):
    """
    Compute the micro averaged score for the Precision Recall curve.
    """
    # Convert y to binarized array for micro and macro scores
    y = label_binarize(y, classes=classes)
    if len(classes) == 2:
        y = np.hstack((1 - y, y))

    # Compute micro-average
    prec, recall, _ = precision_recall_curve(y.ravel(), probas_.ravel())
    ave_prec = average_precision_score(y, probas_, average="micro")
    prec, recall = prec[::-1], recall[::-1]
    prec_rec_auc = auc(recall, prec)
    return prec, recall, prec_rec_auc, ave_prec
