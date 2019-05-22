#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""NSMF model train"""


from __future__ import division
from sklearn import metrics
import numpy as np
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
    y = (np.linalg.norm(user_vector) * np.linalg.norm(item_vector))
    if x == 0:
        return 0
    else:
        return x / y


def train(train_data, test_data, F, alpha, beta, step):
    """
    Args:
        train_data: train data for NSFM
        test_data: test data for NSFM
        F: user vector size, item vector size
        alpha: regularization factor
        beta: learning rate
        step: iteration
    Return:
        {itemid: np.ndarray([v1, v2, v3])}
        {userid: np.ndarray([v1, v2, v3])}
    """
    user_action = get_user_action(train_data)
    item_action = get_item_action(train_data)

    u_t = 0.08 * user_action[0][1]
    i_t = 0.08 * item_action[0][1]
    user_action_dict = {x[0]: x[1] for x in user_action}
    item_action_dict = {x[0]: x[1] for x in item_action}
    u_m = np.mean(np.array(user_action_dict.values()))
    i_m = np.mean(np.array(item_action_dict.values()))

    fp = open("NSMF_result.txt", "wb")

    user_vector, item_vector = {}, {}
    for s in range(step):
        for instance in tqdm(train_data):
            userid, itemid, label = instance
            if userid not in user_vector:
                user_vector[userid] = init(F)
            if itemid not in item_vector:
                item_vector[itemid] = init(F)
            delta = label - predict(user_vector[userid], item_vector[itemid])
            if userid not in user_action_dict:
                user_action_dict[userid] = u_m
            if itemid not in item_action_dict:
                item_action_dict[itemid] = i_m
            u_d = (u_t / (user_action_dict[userid] + u_m)) ** 0.5
            i_d = (i_t / (item_action_dict[itemid] + i_m)) ** 0.5
            for i in range(F):
                user_vector[userid][i] += beta * u_d * (delta*item_vector[itemid][i] - alpha * user_vector[userid][i])
                item_vector[itemid][i] += beta * i_d * (delta*user_vector[userid][i] - alpha * item_vector[itemid][i])
        beta *= 0.9

        fp.write("Train Epochs %d: %s\n" % (s, evaluate(train_data, user_vector, item_vector)))
        fp.write("Test Epochs %d: %s\n" % (s, evaluate(test_data, user_vector, item_vector)))

    fp.close()

    return user_vector, item_vector


def evaluate(test_data, user_vector, item_vector):
    """
    evaluate
    Args:
        test_data: [(userid, itemid, label), ...]
        user_vector: {userid: np.ndarray([v1, v2, v3])}
        item_vector: {itemid: np.ndarray([v1, v2, v3])}
    Return:
        {"AUC": v1, "LogLoss": v2, "RMSE": v3}
    """
    labels, predicts = [], []
    for userid, itemid, label in test_data:
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
    sampling_train_data_path = "ml_100k_sampling_train_data.dat"
    if not os.path.exists(sampling_train_data_path):
        sampling_train_data(train_data, sampling_train_data_path, 1, 4)
    train_data_sampling = []
    with open(sampling_train_data_path, "rb") as fp:
        for line in fp:
            userid, itemid, label = line.strip().split()
            train_data_sampling.append((userid, itemid, float(label)))
    F, alpha, beta, step = 300, 0.01, 0.1, 100
    user_vector, item_vector = train(train_data_sampling, test_data, F, alpha, beta, step)


if __name__ == "__main__":
    main()