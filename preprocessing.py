#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""preprocessing dataset"""


from __future__ import division
import os
import random
from tqdm import tqdm


def get_label_data(path):
    """
    get label data
    Args:
        path: file path
    Return:
        label_data: [(userid, itemid, label), ...]
    """
    if not os.path.exists(path):
        return []

    label_data = []
    with open(path, "rb") as fp:
        for line in fp:
            userid, itemid, rating, timestamp = line.strip().split()
            if float(rating) >= 3.0:
                label_data.append((userid, itemid, 1))
            else:
                label_data.append((userid, itemid, 0))

    return label_data


def get_user_action(label_data):
    """
    get user action
    Args:
        label_data: [(userid, itemid, label), ...]
    Return:
         user_action: [(userid, action_number, [itemid, ...]), ...]
    """
    user_action_dict = {}
    for userid, itemid, label in label_data:
        if userid not in user_action_dict:
            user_action_dict[userid] = []
        if label == 1:
            user_action_dict[userid].append(itemid)

    user_action = []
    for userid, itemids in user_action_dict.items():
        user_action.append((userid, len(itemids), itemids))

    return sorted(user_action, key=lambda x: x[1], reverse=True)


def get_item_action(label_data):
    """
    get item action
    Args:
        label_data: [(userid, itemid, label), ...]
    Return:
         item_action: [(itemid, action_number, [userid, ...]), ...]
    """
    item_action_dict = {}
    for userid, itemid, label in label_data:
        if itemid not in item_action_dict:
            item_action_dict[itemid] = []
        if label == 1:
            item_action_dict[itemid].append(userid)

    item_action = []
    for itemid, userids in item_action_dict.items():
        item_action.append((itemid, len(userids), userids))

    return sorted(item_action, key=lambda x: x[1], reverse=True)


def get_user_no_action(label_data):
    """
    get user no action
    Args:
        label_data: [(userid, itemid, label), ...]
    Return:
         user_no_action: [(userid, no_action_number, [itemid, ...]), ...]
    """
    user_no_action_dict = {}
    for userid, itemid, label in label_data:
        if userid not in user_no_action_dict:
            user_no_action_dict[userid] = []
        if label == 0:
            user_no_action_dict[userid].append(itemid)

    user_no_action = []
    for userid, itemids in user_no_action_dict.items():
        user_no_action.append((userid, len(itemids), itemids))

    return sorted(user_no_action, key=lambda x: x[1], reverse=True)


def get_item_no_action(label_data):
    """
    get item no action
    Args:
        label_data: [(userid, itemid, label), ...]
    Return:
         item_no_action: [(itemid, no_action_number, [userid, ...]), ...]
    """
    item_no_action_dict = {}
    for userid, itemid, label in label_data:
        if itemid not in item_no_action_dict:
            item_no_action_dict[itemid] = []
        if label == 0:
            item_no_action_dict[itemid].append(userid)

    item_no_action = []
    for itemid, userids in item_no_action_dict.items():
        item_no_action.append((itemid, len(userids), userids))

    return sorted(item_no_action, key=lambda x: x[1], reverse=True)


def split_train_test_data(label_data, frac):
    """
    split train test data
    Args:
        label_data: [(userid, itemid, label), ...]
        frac: proportion of training set
    Return:
        train_data: [(userid, itemid, label), ...]
        test_data: [(userid, itemid, label), ...]
    """
    total_size = len(label_data)
    train_size = int(total_size * frac)
    test_size = total_size - train_size

    user_action_dict = {}
    for userid, itemid, label in label_data:
        if userid not in user_action_dict:
            user_action_dict[userid] = ([], [])
        if label == 1:
            user_action_dict[userid][0].append(itemid)
        else:
            user_action_dict[userid][1].append(itemid)

    user_action = []
    for userid, itemids in user_action_dict.items():
        user_action.append((userid, len(itemids[0]), itemids[0], len(itemids[1]), itemids[1]))

    user_size = len(user_action)
    user_test_size = int(test_size / user_size)
    train_data, test_data = [], []
    for userid, positive_number, positive_itemids, negative_number, negative_itemids in user_action:
        l = int(int(user_test_size/2))
        if l < positive_number and l < negative_number:
            sample_number = l
        else:
            sample_number = int((1-frac) * min(positive_number, negative_number))
        user_positive_test_itemids = random.sample(positive_itemids, sample_number)
        user_positive_train_itemids = list(set(positive_itemids).difference(set(user_positive_test_itemids)))
        user_negative_test_itemids = random.sample(negative_itemids, sample_number)
        user_negative_train_itemids = list(set(negative_itemids).difference(set(user_negative_test_itemids)))
        for itemid in user_positive_test_itemids:
            test_data.append((userid, itemid, 1))
        for itemid in user_positive_train_itemids:
            train_data.append((userid, itemid, 1))
        for itemid in user_negative_test_itemids:
            test_data.append((userid, itemid, 0))
        for itemid in user_negative_train_itemids:
            train_data.append((userid, itemid, 0))

    return train_data, test_data


def subsampling_user(user_action, t=0.08, n=100):
    """
    get a user by subsampling
    Args:
        user_action: [(userid, action_numbers, [itemid1,...]), ....]
        t:
        n:
    Return:
        a userid
    """
    if user_action:
        t = t * user_action[0][1]
        p = random.uniform(0, 1)
        d, u = 1.0, -1
        f, d_u = 0, []
        while p < d and f < n:
            f += 1
            u = random.randint(0, len(user_action)-1)
            d = 1 - (t/user_action[u][1]) ** 0.5
            d_u.append((d, u))
        if f == n:
            u = sorted(d_u, key=lambda x: x[0])[0][1]
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
        f, d_i = 0, []
        while p < d and f < 100:
            f += 1
            i = random.randint(0, len(item_action)-1)
            d = 1 - (t/(item_action[i][1])) ** 0.5
            d_i.append((d, i))
        if f == 100:
            i = sorted(d_i, key=lambda x: x[0])[0][1]
        return item_action[i]


def sampling_train_data(train_data, path, n, k):
    """
    get sampling train data
    Args:
        train_data: train data
        path: sampling train data file path
        n: sampling train data size len(train_data)*n*k
        k: negative sampling numbers
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


def main():
    path = "dataset/ml-100k/u.data"
    label_data = get_label_data(path)
    train_data, test_data = split_train_test_data(label_data, 0.8)
    sampling_train_data(train_data, "ml_100k_sampling_train_data.dat", 1, 4)


if __name__ == "__main__":
    main()