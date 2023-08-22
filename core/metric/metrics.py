import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

from utils.registery import METRIC_REGISTRY


def one_hot_transfer(label, class_num):
    """transform to one hot encoding

    Args:
        label (list(int)): label list
        class_num (int): class number

    Returns:
        one-hot-style encoded array
    """
    return np.eye(class_num)[label]


@METRIC_REGISTRY.register()
class ExprMetric:
    def __call__(self, pred, gt, class_num=8):
        """calc expr metric

        Args:
            pred (list(list(int))): 2-d pred class list
            gt (list(list(int))): 2-d label class list
            class_num (int): expr class number (default: 8)

        Returns:
            F1_mean (float): macro expr F1 score
            acc (float): accuracy
            F1 (list(float)): each expr's F1 score
        """
        pred = pred.flatten().tolist()
        gt = gt.flatten().tolist()
        acc = accuracy_score(gt, pred)

        gt = one_hot_transfer(gt, class_num)
        pred = one_hot_transfer(pred, class_num)
        F1 = []
        acc_list = []
        for i in range(class_num):
            gt_ = gt[:, i]
            pred_ = pred[:, i]
            acc_list.append(accuracy_score(gt_, pred_))
            F1.append(f1_score(gt_, pred_))
        F1_mean = np.mean(F1)

        return {'F1': F1_mean, 'ACC': acc, 'F1_list': F1, 'ACC_list': acc_list}
