import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score, recall_score

def metrics(y, score):
    y, score = np.asarray(y), np.asarray(score)
    pred = score >= .5
    return dict(n_patients=len(y), accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                roc_auc=float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
                pr_auc_average_precision=float(average_precision_score(y, score)) if len(set(y)) == 2 else None,
                recall_pos=float(recall_score(y, pred, pos_label=1, zero_division=0)),
                recall_neg=float(recall_score(y, pred, pos_label=0, zero_division=0)), threshold=.5)


