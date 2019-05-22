#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Wide&Deep model"""


import re
import argparse
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from keras import backend
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Reshape, Concatenate, Flatten, Lambda, concatenate
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.regularizers import l2, l1_l2
from preprocessing import *


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


LR = 0.01
BATCH_SIZE = 64
EPOCHS = 100
CATEGORICAL_EMBEDDING_SIZE = 8
R = 1e-3
DOCUMENT_EMBEDDING_SIZE = 300
DOCUMENT_VECTOR_SIZE = 300
BINARY_EMBEDDING_SIZE = 8
K = 300
L1 = 0.01
L2 = 0.01


class IntervalEvaluation(Callback):
    def __init__(self, training_data=(), validation_data=(), interval=1, path="Wide&Deep_result.txt"):
        super(Callback, self).__init__()
        self.interval = interval
        self.path = path
        self.X_train, self.y_train = training_data
        self.X_val, self.y_val = validation_data
        self.fp = open(self.path, "wb")

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            y_train_label = np.array(self.y_train).flatten()
            y_train_pred = np.array(self.model.predict(self.X_train, verbose=0)).flatten()
            y_test_label = np.array(self.y_val).flatten()
            y_test_pred = np.array(self.model.predict(self.X_val, verbose=0)).flatten()

            train_auc = metrics.roc_auc_score(y_train_label, y_train_pred)
            train_logloss = metrics.log_loss(y_train_label, y_train_pred)
            train_rmse = metrics.mean_squared_error(y_train_label, y_train_pred) ** 0.5
            test_auc = metrics.roc_auc_score(y_test_label, y_test_pred)
            test_logloss = metrics.log_loss(y_test_label, y_test_pred)
            test_rmse = metrics.mean_squared_error(y_test_label, y_test_pred) ** 0.5

            self.fp.write("Train Epochs %d: %s\n"
                          % (epoch, {"AUC": train_auc, "LogLoss": train_logloss, "RMSE": train_rmse}))
            self.fp.write("Test Epochs %d: %s\n"
                          % (epoch, {"AUC": test_auc, "LogLoss": test_logloss, "RMSE": test_rmse}))

    def on_train_end(self, logs=None):
        self.fp.close()


def get_data_df(data, user_info_path, item_info_path):
    """
    get data dataframe
    Args:
        data: [(userid, itemid, label), ...]
        user_info_path: user info file path
        item_info_path: item info file path
    Return:
        pd.DataFrame
    """
    du = {"userid": str, "age": str}
    di = {"itemid": str}
    label_df = pd.DataFrame(data, columns=["userid", "itemid", "label"])
    label_df[["userid", "itemid"]] = label_df[["userid", "itemid"]].astype(str)
    user_df = pd.read_csv(user_info_path, sep='|', names=['userid', 'age', 'sex', 'occupation', 'zipcode'], dtype=du)
    item_df = pd.read_csv(item_info_path, sep='|', names=['itemid', 'title', 'date',
                                                          'video_date', 'url', 'is_unknown',
                                                          'is_action', 'is_adventure', 'is_animation',
                                                          'is_children', 'is_comedy', 'is_crime',
                                                          'is_documentary', 'is_drame', 'is_fantasy',
                                                          'is_filmnoir', 'is_horror', 'is_musical',
                                                          'is_mystery', 'is_romance', 'is_scifi',
                                                          'is_thriller', 'is_war', 'is_western'], dtype=di)

    item_df.drop(['video_date', 'url'], axis=1, inplace=True)
    item_df.fillna('24-Sep-1993', inplace=True)

    return label_df.merge(user_df, on="userid").merge(item_df, on="itemid")


def split_train_test_df(df_data, frac):
    """
    split train test dataframe
    Args:
        df_data: pd.DataFrame
        frac: train data proportion
    Return:
        df_train, df_test
    """
    df_train = df_data.sample(frac=frac, random_state=1)
    df_test = df_data.loc[df_data.index.difference(df_train.index)]
    return df_train, df_test


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


def wide(df_train, df_test, wide_cols, cross_cols, document_cols, label, model):
    """
    wide component linear model
    Args:
        df_train: train dataframe
        df_test: test dataframe
        wide_cols: wide columns
        cross_cols: cross columns
        document_cols: document columns
        label: label columns
        model: model type wide deep wide&deep
    Return:
        X_train, y_train, X_test, y_test
    """
    paths = ["ml-100k-wide-deep/wide-x-train.npz",
             "ml-100k-wide-deep/wide-y-train.npz",
             "ml-100k-wide-deep/wide-x-test.npz",
             "ml-100k-wide-deep/wide-y-test.npz"]

    if os.path.exists(paths[0]) and os.path.exists(paths[1]) and os.path.exists(paths[2]) and os.path.exists(paths[3]):
        X_train = load_npz(paths[0]).todense()
        y_train = load_npz(paths[1]).todense()
        X_test = load_npz(paths[2]).todense()
        y_test = load_npz(paths[3]).todense()
    else:
        df_train["is_train"] = 1
        df_test["is_train"] = 0
        df_data = pd.concat([df_train, df_test])
        df_data = cut_word(df_data, document_cols)

        tfidf_columns = []
        for c in document_cols:
            vectorizer = TfidfVectorizer()
            X = np.array(vectorizer.fit_transform(df_data[c]).todense())
            for i in range(X.shape[1]):
                df_data[c+"_"+str(i)] = X[:, i]
                tfidf_columns.append(c+"_"+str(i))
        df_data.drop(document_cols, axis=1, inplace=True)

        cross_columns_dict = cross_columns(cross_cols)
        wide_cols += cross_columns_dict.keys()
        wide_cols += tfidf_columns
        categorical_columns = df_data.select_dtypes(include=['object']).columns.tolist()

        for k, v in cross_columns_dict.iteritems():
            df_data[k] = df_data[v].apply(lambda x: '-'.join(x), axis=1)

        df_data = df_data[[label] + wide_cols + ['is_train']]

        dummy_cols = [c for c in wide_cols if c in categorical_columns + cross_columns_dict.keys()]
        df_data = pd.get_dummies(df_data, columns=dummy_cols)

        train = df_data[df_data.is_train == 1].drop('is_train', axis=1)
        test = df_data[df_data.is_train == 0].drop('is_train', axis=1)

        X_train = train.values[:, 1:]
        y_train = train.values[:, 0].reshape(-1, 1)
        X_test = test.values[:, 1:]
        y_test = test.values[:, 0].reshape(-1, 1)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        save_npz("ml-100k-wide-deep/wide-x-train.npz", csc_matrix(X_train))
        save_npz("ml-100k-wide-deep/wide-y-train.npz", csc_matrix(y_train))
        save_npz("ml-100k-wide-deep/wide-x-test.npz", csc_matrix(X_test))
        save_npz("ml-100k-wide-deep/wide-y-test.npz", csc_matrix(y_test))

    if model == "wide":
        activation, loss = 'sigmoid', 'binary_crossentropy'

        ie = IntervalEvaluation(training_data=(X_train, y_train),
                                validation_data=(X_test, y_test),
                                interval=1,
                                path="Wide&Deep-Wide_result.txt")

        inputs = Input(shape=(X_train.shape[1],), dtype='float32', name="wide_input")
        outputs = Dense(y_train.shape[1], activation=activation)(inputs)
        wide = Model(inputs=inputs, outputs=outputs)
        wide.compile(Adam(LR), loss=loss)
        wide.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, callbacks=[ie])
    else:
        return X_train, y_train, X_test, y_test


def deep(df_train, df_test, embedding_cols, cross_cols, continuous_cols, document_cols, binary_cols, label, model):
    """
    deep component neural network
    Args:
        df_train: train dataframe
        df_test: test dataframe
        embedding_cols: embedding columns
        cross_cols: cross columns
        continuous_cols: continuous columns
        document_cols: document columns
        binary_cols: binary columns
        label: label columns
        model: model type wide deep wide&deep
    """
    df_train["is_train"] = 1
    df_test["is_train"] = 0
    df_data = pd.concat([df_train, df_test])

    df_data = cut_word(df_data, document_cols)
    cross_columns_dict = cross_columns(cross_cols)
    embedding_cols += cross_columns_dict.keys()

    for k, v in cross_columns_dict.iteritems():
        df_data[k] = df_data[v].apply(lambda x: '-'.join(x), axis=1)

    df_data = label_encoder(df_data, embedding_cols)

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

    deep_cols = embedding_cols + continuous_cols + document_cols + binary_cols

    df_data = df_data[[label] + deep_cols + ['is_train']]

    train = df_data[df_data.is_train == 1].drop('is_train', axis=1)
    test = df_data[df_data.is_train == 0].drop('is_train', axis=1)

    X_train, y_train, X_test, y_test = [], None, [], None
    for c in deep_cols:
        if c in embedding_cols + continuous_cols:
            X_train.append(train[c].values)
        if c in document_cols:
            X_train.append(np.array([x for x in train[c].values]))
    X_train.append(train[binary_cols].values)
    for c in deep_cols:
        if c in embedding_cols + continuous_cols:
            X_test.append(test[c].values)
        if c in document_cols:
            X_test.append(np.array([x for x in test[c].values]))
    X_test.append(test[binary_cols].values)
    y_train = train[label].values.reshape(-1, 1)
    y_test = test[label].values.reshape(-1, 1)

    categorical_embeddings = []
    for c in embedding_cols:
        layer_name = c + '_input'
        n_in = df_data[c].nunique()
        n_out = CATEGORICAL_EMBEDDING_SIZE
        r = R
        input_layer, embedding_layer = categorical_embedding(layer_name, n_in, n_out, r)
        categorical_embeddings.append((input_layer, embedding_layer))
        del (input_layer, embedding_layer)

    continuous_embeddings = []
    for c in continuous_cols:
        layer_name = c + '_input'
        input_layer, embedding_layer = continuous_embedding(layer_name)
        continuous_embeddings.append((input_layer, embedding_layer))
        del (input_layer, embedding_layer)

    document_embeddings = []
    for c in document_cols:
        layer_name = c + '_input'
        n_in = max_words[c]
        n_em = DOCUMENT_EMBEDDING_SIZE
        n_out = DOCUMENT_VECTOR_SIZE
        n_words = num_words[c]
        input_layer, embedding_layer = document_embedding(layer_name, n_in, n_em, n_out, n_words)
        document_embeddings.append((input_layer, embedding_layer))
        del (input_layer, embedding_layer)

    binary_embeddings = []
    layer_name = 'binary_input'
    n_in = len(binary_cols)
    n_out = BINARY_EMBEDDING_SIZE
    input_layer, embedding_layer = binary_embedding(layer_name, n_in, n_out)
    binary_embeddings.append((input_layer, embedding_layer))
    del (input_layer, embedding_layer)

    input_layers = [ce[0] for ce in categorical_embeddings]
    input_layers += [ce[0] for ce in continuous_embeddings]
    input_layers += [de[0] for de in document_embeddings]
    input_layers += [be[0] for be in binary_embeddings]
    embedding_layers = [ce[1] for ce in categorical_embeddings]
    embedding_layers += [ce[1] for ce in continuous_embeddings]
    embedding_layers += [de[1] for de in document_embeddings]
    embedding_layers += [be[1] for be in binary_embeddings]

    if model == "deep":
        activation, loss = 'sigmoid', 'binary_crossentropy'

        ie = IntervalEvaluation(training_data=(X_train, y_train),
                                validation_data=(X_test, y_test),
                                interval=1,
                                path="Wide&Deep-Deep_result.txt")

        x = Concatenate(axis=-1)(embedding_layers)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(K, activation='relu', kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
        x = Dense(K, activation='relu', kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
        x = Dense(K, activation='relu', kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
        x = Dense(y_train.shape[1], activation=activation)(x)
        deep = Model(inputs=input_layers, outputs=x)
        deep.compile(Adam(LR), loss=loss)
        deep.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, callbacks=[ie])
    else:
        return X_train, y_train, X_test, y_test, input_layers, embedding_layers


def wide_deep(df_train,
              df_test,
              wide_cols,
              cross_cols,
              document_cols,
              embedding_cols,
              continuous_cols,
              binary_cols,
              label,
              model):
    """
    wide&deep model
    Args:
        df_train: pd.DataFrame
        df_test: pd.DataFrame
        wide_cols: wide columns
        cross_cols: cross columns
        document_cols: document columns
        embedding_cols: embedding columns
        continuous_cols: continuous columns
        binary_cols: binary columns
        label: label columns
        model: model type wide deep wide&deep
    """
    X_train_wide, y_train_wide, X_test_wide, y_test_wide = \
        wide(df_train, df_test, wide_cols, cross_cols, document_cols, label, model)
    X_train_deep, y_train_deep, X_test_deep, y_test_deep, deep_input_layers, deep_embedding_layers = \
        deep(df_train, df_test, embedding_cols, cross_cols, continuous_cols, document_cols, binary_cols, label, model)

    X_train_wd = [X_train_wide] + X_train_deep
    Y_train_wd = y_train_deep
    X_test_wd = [X_test_wide] + X_test_deep
    Y_test_wd = y_test_deep

    activation, loss = 'sigmoid', 'binary_crossentropy'

    ie = IntervalEvaluation(training_data=(X_train_wd, Y_train_wd),
                            validation_data=(X_test_wd, Y_test_wd),
                            interval=1,
                            path="Wide&Deep_result.txt")

    # wide
    w = Input(shape=(X_train_wide.shape[1],), dtype='float32', name='wide')

    # deep
    d = Concatenate(axis=-1)(deep_embedding_layers)
    d = Flatten()(d)
    d = BatchNormalization()(d)
    d = Dense(K, activation='relu', kernel_regularizer=l1_l2(l1=L1, l2=L2))(d)
    d = Dense(K, activation='relu', kernel_regularizer=l1_l2(l1=L1, l2=L2))(d)
    d = Dense(K, activation='relu', kernel_regularizer=l1_l2(l1=L1, l2=L2))(d)

    # wide&deep
    wd = concatenate([w, d])
    out = Dense(Y_train_wd.shape[1], activation=activation, name='wide_deep')(wd)
    wide_deep = Model(inputs=[w] + deep_input_layers, outputs=out)
    wide_deep.compile(Adam(LR), loss=loss)
    wide_deep.fit(X_train_wd, Y_train_wd, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, callbacks=[ie])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="wide&deep", help="wide, deep or both")
    args = vars(parser.parse_args())
    model = args["model"]

    path = "dataset/ml-100k/u.data"
    label_data = get_label_data(path)
    user_info_path = "dataset/ml-100k/u.user"
    item_info_path = "dataset/ml-100k/u.item"
    train_data, test_data = split_train_test_data(label_data, 0.8)
    df_train = get_data_df(train_data, user_info_path, item_info_path)
    df_test = get_data_df(test_data, user_info_path, item_info_path)

    wide_cols = ['age', 'sex', 'occupation', 'date', 'is_unknown',
                 'is_action', 'is_adventure', 'is_animation',
                 'is_children', 'is_comedy', 'is_crime',
                 'is_documentary', 'is_drame', 'is_fantasy',
                 'is_filmnoir', 'is_horror', 'is_musical',
                 'is_mystery', 'is_romance', 'is_scifi',
                 'is_thriller', 'is_war', 'is_western']
    cross_cols = [['age', 'sex'],
                  ['age', 'occupation']]
    document_cols = ['title']

    embedding_cols = ['userid', 'itemid', 'zipcode', 'age', 'sex', 'occupation', 'date']

    continuous_cols = []
    binary_cols = ['is_action', 'is_adventure', 'is_animation',
                 'is_children', 'is_comedy', 'is_crime',
                 'is_documentary', 'is_drame', 'is_fantasy',
                 'is_filmnoir', 'is_horror', 'is_musical',
                 'is_mystery', 'is_romance', 'is_scifi',
                 'is_thriller', 'is_war', 'is_western']
    label = 'label'

    if model == "wide":
        wide(df_train,
             df_test,
             wide_cols,
             cross_cols,
             document_cols,
             label,
             model)
    elif model == "deep":
        deep(df_train,
             df_test,
             embedding_cols,
             cross_cols,
             continuous_cols,
             document_cols,
             binary_cols,
             label,
             model)
    else:
        wide_deep(df_train,
                  df_test,
                  wide_cols,
                  cross_cols,
                  document_cols,
                  embedding_cols,
                  continuous_cols,
                  binary_cols,
                  label,
                  model)


if __name__ == "__main__":
    main()