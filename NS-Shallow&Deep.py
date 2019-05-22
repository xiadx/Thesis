#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""NS-Shallow&Deep model"""


import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils import plot_model
from keras import backend
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Reshape, Concatenate, Flatten, Lambda, concatenate, Activation
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import dot
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.regularizers import l2, l1_l2
from preprocessing import *


def ns_train_data(train_data, n, k, path):
    """
    negative sampling train data
    Args:
        train_data: train data
        n: sampling train data size len(train_data)*n
        k: negative sampling numbers
        path: file path
    Return:
        ns_train_data: [[userid, pitemid, n1itemid, n2itemid, ..., nkitemid], ...]
    """
    ntd = []

    if not os.path.exists(path):
        user_action = get_user_action(train_data)
        item_action = get_item_action(train_data)
        user_no_action = get_user_no_action(train_data)
        item_action_dict = {x[0]: x for x in item_action}
        user_no_action_dict = {x[0]: x for x in user_no_action}
        total_itemids = set([x[0] for x in item_action])

        l = int(len(train_data) * n)
        for i in tqdm(range(l)):
            userid, action_number, itemids = subsampling_user(user_action)
            postive_item_action = [item_action_dict[itemid] for itemid in itemids]
            sort_postive_item_action = sorted(postive_item_action, key=lambda x: x[1], reverse=True)
            postive_itemid = subsampling_item(sort_postive_item_action)[0]
            negative_itemids = user_no_action_dict[userid][2]
            if not negative_itemids:
                negative_itemids = list(total_itemids.difference(set(itemids)))
            negative_k_itemids = [random.choice(negative_itemids) for _ in range(k)]
            ntd.append([userid, postive_itemid] + negative_k_itemids)

        with open(path, "wb") as fp:
            for x in ntd:
                fp.write(" ".join(x) + "\n")
    else:
        with open(path, "rb") as fp:
            for line in fp:
                ntd.append(line.strip().split())

    return ntd


def get_user_df(path):
    """
    get user dataframe
    Args:
        path: file path
    Return:
        pd.DataFrame
    """
    if not os.path.exists(path):
        return
    du = {"userid": str, "age": str}
    df_user = pd.read_csv(path, sep='|', names=['userid', 'age', 'sex', 'occupation', 'zipcode'], dtype=du)
    return df_user


def get_item_df(path):
    """
    get item dataframe
    Args:
        path: file path
    Return:
        pd.Dataframe
    """
    if not os.path.exists(path):
        return
    di = {"itemid": str}
    df_item = pd.read_csv(path, sep='|', names=['itemid', 'title', 'date',
                                                'video_date', 'url', 'is_unknown',
                                                'is_action', 'is_adventure', 'is_animation',
                                                'is_children', 'is_comedy', 'is_crime',
                                                'is_documentary', 'is_drame', 'is_fantasy',
                                                'is_filmnoir', 'is_horror', 'is_musical',
                                                'is_mystery', 'is_romance', 'is_scifi',
                                                'is_thriller', 'is_war', 'is_western'], dtype=di)

    df_item.drop(['video_date', 'url'], axis=1, inplace=True)
    df_item.fillna('24-Sep-1993', inplace=True)
    return df_item


def cut_word(df_data, document_cols):
    """
    cut word
    Args:
        df_data: pd.DataFrame
        document_cols: document columns
    Return:
        pd.DataFrame
    """
    stopwords = '!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n'

    def deal(document):
        """
        deal a document
        """
        pattern = re.compile(r'[^\x00-\x7f]')
        return " ".join("".join([c for c in re.sub(pattern, '', document) if c not in stopwords]).split())

    for c in document_cols:
        df_data[c] = df_data[c].apply(deal)

    return df_data


def cross_columns(cross_cols):
    """
    cross columns
    Args:
        cross_cols: [[c1, c2], [c3, c4, c5], ...]
    Return:
        {c1_c2: [c1, c2], c3_c4_c5: [c3, c4, c5], ...}
    """
    cross_columns_dict = {}
    cols = ['_'.join(cc) for cc in cross_cols]
    for c, cc in zip(cols, cross_cols):
        cross_columns_dict[c] = cc
    return cross_columns_dict


def label_encoder(df_data, label_cols):
    """
    label encoder
    Args:
        df_data: pd.DataFrame
        label_cols: label columns
    """
    le = LabelEncoder()
    for c in label_cols:
        df_data[c] = le.fit_transform(df_data[c].values)
    return df_data


def categorical_embedding(name, n_in, n_out, r):
    """
    categorical input embedding
    Args:
        name: layer name
        n_in: input dimension
        n_out: output dimension
        r: regularization parameter
    Return:
        (input_layer, embedding_layer)
    """
    input_layer = Input(shape=(1,), dtype='int64', name=name)
    embedding_layer = Embedding(n_in, n_out, input_length=1, embeddings_regularizer=l2(r))(input_layer)
    return input_layer, embedding_layer


def continuous_embedding(name):
    """
    continuous input embedding
    Args:
        name: layer name
    Return:
        (input_layer, embedding_layer)
    """
    input_layer = Input(shape=(1,), dtype='float32', name=name)
    embedding_layer = Reshape((1, 1))(input_layer)
    return input_layer, embedding_layer


def document_embedding(name, n_in, n_em, n_out, n_words):
    """
    document input embedding
    Args:
         name: layer name
         n_in: input max numbers of words
         n_em: embedding size
         n_out: output dimension
         n_words: num words
    Return:
        (input_layer, reshape_layer
    """
    input_layer = Input(shape=(n_in,), name=name)
    embedding_layer = Embedding(n_words, n_em)(input_layer)
    convolution_layer = Convolution1D(n_out, 1, padding="same", input_shape=(n_in, n_em), activation="tanh")(embedding_layer)
    pooling_layer = Lambda(lambda x: backend.max(x, axis=1), output_shape=(n_out,))(convolution_layer)
    reshape_layer = Reshape((1, n_out))(pooling_layer)
    return input_layer, reshape_layer


def binary_embedding(name, n_in, n_out):
    """
    binary embedding
    Args:
        name: layer name
        n_in: input dimension
        n_out: output dimension
    Return:
        input_layer, reshape_layer
    """
    input_layer = Input(shape=(n_in,), dtype='float32', name=name)
    dense_layer = Dense(n_out, activation="relu")(input_layer)
    reshape_layer = Reshape((1, n_out))(dense_layer)
    return input_layer, reshape_layer


def wide_feature(df_data, categorical_cols, cross_cols, document_cols, k):
    """
    wide feature
    Args:
        df_user: pd.DataFrame
        categorical_cols: categorical columns
        cross_clos: cross columns
        document_cols: document columns
        k: "userid" or "itemid"
    Return:
        {userid: np.array([v1, v2]), ...}
    """
    ids = df_data[k].values

    df_data = cut_word(df_data, document_cols)
    tfidf_columns = []
    for c in document_cols:
        vectorizer = TfidfVectorizer()
        X = np.array(vectorizer.fit_transform(df_data[c]).todense())
        for i in range(X.shape[1]):
            df_data[c + "_" + str(i)] = X[:, i]
            tfidf_columns.append(c + "_" + str(i))
    df_data.drop(document_cols, axis=1, inplace=True)

    cross_columns_dict = cross_columns(cross_cols)
    for k, v in cross_columns_dict.iteritems():
        df_data[k] = df_data[v].apply(lambda x: '-'.join(x), axis=1)

    dummy_cols = categorical_cols + cross_columns_dict.keys()
    df_data = pd.get_dummies(df_data, columns=dummy_cols)

    scaler = MinMaxScaler()
    np_data = scaler.fit_transform(df_data.values)

    return {ids[i]: np_data[i] for i in range(len(ids))}


def deep_feature(df_data, categorical_cols, cross_cols, document_cols, k):
    """
    deep feature
    Args:
        df_user: pd.DataFrame
        categorical_cols: categorical columns
        continuous_cols: continuous columns
        document_cols: documents columns
        bianry_cols: binary columns
        k: userid columns
    Return:
        {userid: {col: np.array}}
    """
    ids = df_data[k].values

    df_data = cut_word(df_data, document_cols)
    cross_columns_dict = cross_columns(cross_cols)
    categorical_cols += cross_columns_dict.keys()

    for k, v in cross_columns_dict.iteritems():
        df_data[k] = df_data[v].apply(lambda x: '-'.join(x), axis=1)

    df_data = label_encoder(df_data, categorical_cols)

    max_words = {}
    num_words = {}
    for c in document_cols:
        texts = df_data[c].values
        max_words[c] = max([len(text.split()) for text in texts])
        num_words[c] = len(set([t for text in texts for t in text]))

    for c in document_cols:
        texts = df_data[c].values
        tk = Tokenizer(num_words=num_words[c], lower=True)
        tk.fit_on_texts(texts)
        df_data[c] = pad_sequences(tk.texts_to_sequences(texts), maxlen=max_words[c]).tolist()

    for c in categorical_cols:
        print df_data[c].nunique()

    deep_feature = defaultdict(lambda: {})
    for c in df_data.columns:
        features = df_data[c].values
        for i in range(len(ids)):
            deep_feature[ids[i]][c] = features[i]

    return deep_feature


def model_train_data(ns_train_data,
                     user_wide_feature,
                     user_deep_feature,
                     item_wide_feature,
                     item_deep_feature):
    """
    get model train data
    Args:
        ns_train_data: [userid, pitemid, n1itemid, ..., nkitemid]
        user_wide_feature: {userid: np.array([v1, v2, ...])}
        user_deep_feature: {userid: {col: value, ...}, ...}
        item_wide_feature: {itemid: np.array([v1, v2, ...])}
        item_deep_feature: {itemid: {col: value, ...}, ...}
    Return:
        [[uwi, udi1, udi2, ...], [iwi, idi1, ...], [], [], [], []]
    """
    k = len(ns_train_data[0])
    uwi = []
    for x in ns_train_data:
        uwi.append(user_wide_feature[x[0]])
        user_deep_feature[x[0]]

    print ns_train_data
    print user_wide_feature
    print user_deep_feature
    print item_wide_feature
    print item_deep_feature


uwi = Input(shape=(2359,), name="uwi")
udc = [Input(shape=(1,), name="udi_" + str(i)) for i in range(7)]
iwi1 = Input(shape=(4373,), name="iwi1")
idc1 = [Input(shape=(1,), name="idc1_" + str(i)) for i in range(2)]
idd1 = Input(shape=(16,), name="idd1")
idb1 = Input(shape=(20,), name="idb1")
iwi2 = Input(shape=(4373,), name="iwi2")
idc2 = [Input(shape=(1,), name="idc2_" + str(i)) for i in range(2)]
idd2 = Input(shape=(16,), name="idd2")
idb2 = Input(shape=(20,), name="idb2")
iwi3 = Input(shape=(4373,), name="iwi3")
idc3 = [Input(shape=(1,), name="idc3_" + str(i)) for i in range(2)]
idd3 = Input(shape=(16,), name="idd3")
idb3 = Input(shape=(20,), name="idb3")
iwi4 = Input(shape=(4373,), name="iwi4")
idc4 = [Input(shape=(1,), name="idc4_" + str(i)) for i in range(2)]
idd4 = Input(shape=(16,), name="idd4")
idb4 = Input(shape=(20,), name="idb4")
iwi5 = Input(shape=(4373,), name="iwi5")
idc5 = [Input(shape=(1,), name="idc5_" + str(i)) for i in range(2)]
idd5 = Input(shape=(16,), name="idd5")
idb5 = Input(shape=(20,), name="idb5")


def ufv():
    ci = [943, 61, 2, 21, 795, 429, 108]
    udce = [Embedding(ci[i], 8, input_length=1, embeddings_regularizer=l2(0.01))(udc[i]) for i in range(7)]

    x = Concatenate(axis=-1)(udce)
    x = Flatten()(x)
    x = BatchNormalization()(x)

    uwd = concatenate([uwi, x])

    x = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(uwd)
    x = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(uwd)
    ufv = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(uwd)

    return ufv


def ifv1():
    ii = [1682, 241]
    idce = [Embedding(ii[i], 8, input_length=1, embeddings_regularizer=l2(0.01))(idc1[i]) for i in range(2)]
    idde = Embedding(64, 300)(idd1)
    iddecl = Convolution1D(300, 1, padding="same", input_shape=(16, 300), activation="tanh")(idde)
    iddepl = Lambda(lambda x: backend.max(x, axis=1), output_shape=(300,))(iddecl)
    idderl = Reshape((1, 300))(iddepl)

    idbdl = Dense(8, activation="relu")(idb1)
    idbrl = Reshape((1, 8))(idbdl)

    y = Concatenate(axis=-1)(idce + [idderl] + [idbrl])
    y = Flatten()(y)
    y = BatchNormalization()(y)

    iwd = concatenate([iwi1, y])

    y = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)
    y = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)
    ifv = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)

    return ifv


def ifv2():
    ii = [1682, 241]
    idce = [Embedding(ii[i], 8, input_length=1, embeddings_regularizer=l2(0.01))(idc2[i]) for i in range(2)]
    idde = Embedding(64, 300)(idd2)
    iddecl = Convolution1D(300, 1, padding="same", input_shape=(16, 300), activation="tanh")(idde)
    iddepl = Lambda(lambda x: backend.max(x, axis=1), output_shape=(300,))(iddecl)
    idderl = Reshape((1, 300))(iddepl)

    idbdl = Dense(8, activation="relu")(idb2)
    idbrl = Reshape((1, 8))(idbdl)

    y = Concatenate(axis=-1)(idce + [idderl] + [idbrl])
    y = Flatten()(y)
    y = BatchNormalization()(y)

    iwd = concatenate([iwi2, y])

    y = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)
    y = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)
    ifv = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)

    return ifv


def ifv3():
    ii = [1682, 241]
    idce = [Embedding(ii[i], 8, input_length=1, embeddings_regularizer=l2(0.01))(idc3[i]) for i in range(2)]
    idde = Embedding(64, 300)(idd3)
    iddecl = Convolution1D(300, 1, padding="same", input_shape=(16, 300), activation="tanh")(idde)
    iddepl = Lambda(lambda x: backend.max(x, axis=1), output_shape=(300,))(iddecl)
    idderl = Reshape((1, 300))(iddepl)

    idbdl = Dense(8, activation="relu")(idb3)
    idbrl = Reshape((1, 8))(idbdl)

    y = Concatenate(axis=-1)(idce + [idderl] + [idbrl])
    y = Flatten()(y)
    y = BatchNormalization()(y)

    iwd = concatenate([iwi3, y])

    y = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)
    y = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)
    ifv = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)

    return ifv


def ifv4():
    ii = [1682, 241]
    idce = [Embedding(ii[i], 8, input_length=1, embeddings_regularizer=l2(0.01))(idc4[i]) for i in range(2)]
    idde = Embedding(64, 300)(idd3)
    iddecl = Convolution1D(300, 1, padding="same", input_shape=(16, 300), activation="tanh")(idde)
    iddepl = Lambda(lambda x: backend.max(x, axis=1), output_shape=(300,))(iddecl)
    idderl = Reshape((1, 300))(iddepl)

    idbdl = Dense(8, activation="relu")(idb4)
    idbrl = Reshape((1, 8))(idbdl)

    y = Concatenate(axis=-1)(idce + [idderl] + [idbrl])
    y = Flatten()(y)
    y = BatchNormalization()(y)

    iwd = concatenate([iwi4, y])

    y = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)
    y = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)
    ifv = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)

    return ifv


def ifv5():
    ii = [1682, 241]
    idce = [Embedding(ii[i], 8, input_length=1, embeddings_regularizer=l2(0.01))(idc5[i]) for i in range(2)]
    idde = Embedding(64, 300)(idd5)
    iddecl = Convolution1D(300, 1, padding="same", input_shape=(16, 300), activation="tanh")(idde)
    iddepl = Lambda(lambda x: backend.max(x, axis=1), output_shape=(300,))(iddecl)
    idderl = Reshape((1, 300))(iddepl)

    idbdl = Dense(8, activation="relu")(idb5)
    idbrl = Reshape((1, 8))(idbdl)

    y = Concatenate(axis=-1)(idce + [idderl] + [idbrl])
    y = Flatten()(y)
    y = BatchNormalization()(y)

    iwd = concatenate([iwi5, y])

    y = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)
    y = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)
    ifv = Dense(300, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(iwd)

    return ifv


def shallow_deep():
    """
    shallow&deep model
    """


    inputs = [uwi] + udc + \
             [iwi1]+ idc1+ [idd1] + [idb1] + \
             [iwi2]+ idc2 + [idd2] + [idb2] + \
             [iwi3] + idc3 + [idd3] + [idb3] + \
             [iwi4] + idc4 + [idd4] + [idb4] + \
             [iwi5] + idc5 + [idd5] + [idb5]

    ufv1 = ufv()
    ifv_1 = ifv1()
    ifv_2 = ifv2()
    ifv_3 = ifv3()
    ifv_4 = ifv4()
    ifv_5 = ifv5()

    uipp = dot([ufv1, ifv_1], axes=1, normalize=True)
    uinps = [dot([ufv1, ifv_2], axes=1, normalize=True),
             dot([ufv1, ifv_3], axes=1, normalize=True),
             dot([ufv1, ifv_4], axes=1, normalize=True),
             dot([ufv1, ifv_5], axes=1, normalize=True)]
    outputs = concatenate([uipp] + uinps)
    outputs = Reshape((4 + 1, 1))(outputs)

    weight = np.array([1]).reshape(1, 1, 1)
    with_gamma = Convolution1D(1,
                               1,
                               padding="same",
                               input_shape=(4 + 1, 1),
                               activation="linear",
                               use_bias=False,
                               weights=[weight])(outputs)
    with_gamma = Reshape((4 + 1,))(with_gamma)

    prob = Activation("softmax")(with_gamma)

    model = Model(inputs=inputs, outputs=prob)
    model.compile(optimizer="adadelta", loss="categorical_crossentropy")

    plot_model(model, to_file="ns-shallow&deep.png", show_shapes=True)


def main():
    # path = "dataset/ml-100k/u.data"
    # user_info_path = "dataset/ml-100k/u.user"
    # item_info_path = "dataset/ml-100k/u.item"
    # label_data = get_label_data(path)
    # train_data, test_data = split_train_test_data(label_data, 0.8)
    #
    # ntd = ns_train_data(train_data, 1, 4, "ml_100k_ns_train_data.dat")
    # ucc = ["userid", "age", "sex", "occupation", "zipcode"]
    # urc = [["age", "sex"], ["age", "occupation"]]
    # udc = []
    # uk = "userid"
    # icc = ["itemid", "date"]
    # irc = []
    # idc = ["title"]
    # ik = "itemid"
    # df_user = get_user_df(user_info_path)
    # df_item = get_item_df(item_info_path)
    # # uwf = wide_feature(df_user, ucc, urc, udc, uk)
    # # iwf = wide_feature(df_item, icc, irc, idc, ik)
    #
    # # df_user = get_user_df(user_info_path)
    # # df_item = get_item_df(item_info_path)
    # idf = deep_feature(df_item, icc, irc, idc, ik)
    # # udf = deep_feature(df_user, ucc, urc, udc, uk)
    #
    # # model_train_data(ntd, uwf, udf, iwf, idf)
    shallow_deep()


if __name__ == "__main__":
    main()