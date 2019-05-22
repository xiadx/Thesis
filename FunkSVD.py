#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""FunkSVD model train"""


from __future__ import division
import numpy as np
from sklearn import metrics
from preprocessing import *


def init(F):
    """
    Args:
        F: vector size
    Return:
        np.ndarray
    """
    return np.random.randn(F)


def predict(user_vector, item_vector):
    """
    user vector and item vector distance
    Args:
        user_vector: user vector
        item_vector: item vector
    Return:
        user vector and item vector similarity
    """
    x = np.dot(user_vector, item_vector)
    y = (np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
    if x == 0:
        return 0
    else:
        return x / y


def train(train_data, test_data, F, alpha, beta, step):
    """
    Args:
        train_data: train data for FunkSVD
        test_data: test data for FunkSVD
        F: user vector size, item vector size
        alpha: regularization factor
        beta: learning rate
        step: iteration
    Return:
        {itemid: np.ndarray([v1, v2, v3])}
        {userid: np.ndarray([v1, v2, v3])}
    """
    fp = open("FunkSVD_result.txt", "wb")

    user_vector, item_vector = {}, {}
    for s in range(step):
        for instance in tqdm(train_data):
            userid, itemid, label = instance
            if userid not in user_vector:
                user_vector[userid] = init(F)
            if itemid not in item_vector:
                item_vector[itemid] = init(F)
            delta = label - predict(user_vector[userid], item_vector[itemid])
            for i in range(F):
                user_vector[userid][i] += beta * (delta*item_vector[itemid][i] - alpha * user_vector[userid][i])
                item_vector[itemid][i] += beta * (delta*user_vector[userid][i] - alpha * item_vector[itemid][i])
        beta *= 0.9

        fp.write("Train Epochs %d: %s\n" % (s, evaluate(train_data, user_vector, item_vector)))
        fp.write("Test Epochs %d: %s\n" % (s, evaluate(test_data, user_vector, item_vector)))

    fp.close()

    return user_vector, item_vector


def evaluate(data, user_vector, item_vector):
    """
    evaluate
    Args:
        data: [(userid, itemid, label), ...]
        user_vector: {userid: np.ndarray([v1, v2, v3])}
        item_vector: {itemid: np.ndarray([v1, v2, v3])}
    Return:
        {"AUC": v1, "LogLoss": v2, "RMSE": v3}
    """
    labels, predicts = [], []
    for userid, itemid, label in data:
        if userid in user_vector and itemid in item_vector:
            pred = predict(user_vector[userid], item_vector[itemid])
            labels.append(label)
            predicts.append(pred)
    auc = metrics.roc_auc_score(labels, predicts)
    logloss = metrics.log_loss(labels, predicts)
    rmse = metrics.mean_squared_error(labels, predicts) ** 0.5
    return {"AUC": auc, "LogLoss": logloss, "RMSE": rmse}


def main():
    path = "dataset/ml-100k/u.data"
    label_data = get_label_data(path)
    train_data, test_data = split_train_test_data(label_data, 0.8)
    F, alpha, beta, step = 300, 0.01, 0.1, 100
    user_vector, item_vector = train(train_data, test_data, F, alpha, beta, step)


if __name__ == "__main__":
    main()