#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""LDA-NSMF model train"""


from __future__ import division
import numpy as np
from gensim import corpora, models
from gensim.matutils import sparse2full
from sklearn import  metrics
from preprocessing import *


def subsampling_user(user_action):
    """
    get a user by subsampling
    Args:
        user_action: user action
    Return:
        a userid
    """
    if user_action:
        t = 0.08 * user_action[0][1]
        p = random.uniform(0, 1)
        d, u = 1.0, -1
        f = 0
        while p < d and f < 100:
            f += 1
            u = random.randint(0, len(user_action)-1)
            d = 1 - (t/user_action[u][1]) ** 0.5
        return user_action[u]


def subsampling_item(item_action):
    """
    get a item by subsampling
    Args:
        item_action: item action
    Return:
        itemid
    """
    if item_action:
        t = 0.08 * item_action[0][1]
        p = random.uniform(0, 1)
        d, i = 1.0, -1
        f = 0
        while p < d and f < 100:
            f += 1
            i = random.randint(0, len(item_action)-1)
            d = 1 - (t/(item_action[i][1])) ** 0.5
        return item_action[i]


def sampling_train_data(train_data, path, n, k):
    """
    get sampling train data
    Args:
        train_data: train data
        path: sampling train data file path
        n: sampling train data size len(train_data)*n*k
    Return:
        sampling train data
    """
    user_action = get_user_action(train_data)
    item_action = get_item_action(train_data)
    user_no_action = get_user_no_action(train_data)
    item_action_dict = {x[0]: x for x in item_action}
    user_no_action_dict = {x[0]: x for x in user_no_action}
    total_itemids = set([x[0] for x in item_action])

    l = int(len(train_data) * n)
    with open(path, "wb") as fp:
        for i in tqdm(range(l)):
            userid, action_number, itemids = subsampling_user(user_action)
            postive_item_action = [item_action_dict[itemid] for itemid in itemids]
            sort_postive_item_action = sorted(postive_item_action, key=lambda x: x[1], reverse=True)
            postive_itemid = subsampling_item(sort_postive_item_action)[0]
            negative_itemids = user_no_action_dict[userid][2]
            if not negative_itemids:
                negative_itemids = list(total_itemids.difference(set(itemids)))
            negative_k_itemids = [random.choice(negative_itemids) for _ in range(k)]
            fp.write(userid + "\t" + postive_itemid + "\t" + "1\n")
            for negative_itemid in negative_k_itemids:
                fp.write(userid + "\t" + negative_itemid + "\t" + "0\n")


def get_user_latent_vector(user_action, path):
    """
    get user latent vector
    Args:
        user_action: user action
        path: lda model path
    Return:
        user_latent_vector: {userid: np.ndarray([v1, v2, v3])}
    """
    texts = [x[2] for x in user_action]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    if not os.path.exists(path):
        lda = models.ldamodel.LdaModel(corpus=corpus, num_topics=F, id2word=dictionary)
        lda.save(path)
    else:
        lda = models.ldamodel.LdaModel.load(path)
    topics = lda.get_document_topics(corpus)
    user_latent_vector = {user_action[i][0]: sparse2full(topics[i], lda.num_topics) for i in range(len(texts))}
    return user_latent_vector


def get_item_latent_vector(item_action, path):
    """
    get item latent vector
    Args:
        item_action: item action
        path: lda model path
    Return:
        item_latent_vector: {itemid: np.ndarray([v1, v2, v3])}
    """
    texts = [x[2] for x in item_action]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    if not os.path.exists(path):
        lda = models.ldamodel.LdaModel(corpus=corpus, num_topics=F, id2word=dictionary)
        lda.save(path)
    else:
        lda = models.ldamodel.LdaModel.load(path)
    topics = lda.get_document_topics(corpus)
    item_latent_vector = {item_action[i][0]: sparse2full(topics[i], lda.num_topics) for i in range(len(texts))}
    return item_latent_vector


F, alpha, beta, step = 300, 0.01, 0.1, 100
path = "dataset/ml-100k/u.data"
label_data = get_label_data(path)
train_data, test_data = split_train_test_data(label_data, 0.8)
user_action = get_user_action(train_data)
item_action = get_item_action(train_data)
user_latent_vector = get_user_latent_vector(user_action, "ml-100k-lda-model/user_lda_model")
item_latent_vector = get_item_latent_vector(item_action, "ml-100k-lda-model/item_lda_model")


def init(kind, id):
    """
    lda model result init
    Args:
         kind: 'u', 'i' userid, itemid
         id: userid or itemid
    Return
        np.ndarray
    """
    if kind == 'u':
        if id in user_latent_vector:
            return user_latent_vector[id]
        else:
            return np.random.randn(F)
    else:
        if id in item_latent_vector:
            return item_latent_vector[id]
        else:
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
    return np.dot(user_vector, item_vector) / (np.linalg.norm(user_vector)*np.linalg.norm(item_vector))


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
    u_t = 0.08 * user_action[0][1]
    i_t = 0.08 * item_action[0][1]
    user_action_dict = {x[0]: x[1] for x in user_action}
    item_action_dict = {x[0]: x[1] for x in item_action}
    u_m = np.mean(np.array(user_action_dict.values()))
    i_m = np.mean(np.array(item_action_dict.values()))

    fp = open("LDA-NSFM_result.txt", "wb")

    user_vector, item_vector = {}, {}
    for s in range(step):
        for instance in tqdm(train_data):
            userid, itemid, label = instance
            if userid not in user_vector:
                user_vector[userid] = init('u', userid)
            if itemid not in item_vector:
                item_vector[itemid] = init('i', itemid)
            delta = label - predict(user_vector[userid], item_vector[itemid])
            if userid not in user_action_dict:
                user_action_dict[userid] = u_m
            if itemid not in item_action_dict:
                item_action_dict[itemid] = i_m
            u_d = (u_t/(user_action_dict[userid]+u_m)) ** 0.5
            i_d = (i_t/(item_action_dict[itemid]+i_m)) ** 0.5
            for i in range(F):
                user_vector[userid][i] += beta * u_d * (delta*item_vector[itemid][i] - alpha * user_vector[userid][i])
                item_vector[itemid][i] += beta * i_d * (delta*user_vector[userid][i] - alpha * item_vector[itemid][i])
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
    sampling_train_data_path = "ml_100k_sampling_train_data.dat"
    if not os.path.exists(sampling_train_data_path):
        sampling_train_data(train_data, sampling_train_data_path, 5, 4)
    train_data_sampling = []
    with open(sampling_train_data_path, "rb") as fp:
        for line in fp:
            userid, itemid, label = line.strip().split()
            train_data_sampling.append((userid, itemid, float(label)))
    user_vector, item_vector = train(train_data_sampling, test_data, F, alpha, beta, step)


if __name__ == "__main__":
    main()