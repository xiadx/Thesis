#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def nscf_learning_curves():
    train_logloss = [13.9749, 12.1171, 10.8976, 7.2394, 4.3570, 3.8065, 3.7157, 3.2558, 2.9546, 2.7440, 1.5259,
                     0.9440, 0.9199, 0.8933, 0.8482, 0.7984, 0.7800, 0.7760, 0.7566, 0.7202, 0.7153, 0.7126, 0.7015, 0.6928, 0.6875, 0.6818, 0.6764, 0.6648, 0.6625, 0.6584, 0.6416, 0.6307, 0.6237, 0.6235, 0.6231, 0.6167, 0.6102, 0.6044, 0.5916, 0.5734]
    test_logloss = [15.7243, 13.5615, 11.4896, 8.4220, 5.8326, 4.6092, 4.0900, 3.7464, 3.6078, 3.4541, 2.9277, 2.1484,
                    1.6946, 1.4267, 1.3630, 1.2300, 1.1092, 0.9644, 0.9417, 0.9174, 0.8959, 0.8343, 0.8138, 0.7931,
                    0.7735, 0.7510, 0.7341, 0.7249, 0.7147, 0.6913, 0.6716, 0.6557, 0.6357, 0.6235, 0.6287, 0.6125,
                    0.6066, 0.5824, 0.6006, 0.6106]
    x = np.linspace(1, len(train_logloss), len(train_logloss))
    plt.figure(figsize=(6, 3))
    plt.grid(linestyle="--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(x, train_logloss, marker='o', color="SkyBlue", label="Train Data LogLoss", linewidth=1.5)
    plt.plot(x, test_logloss, marker='^', color="IndianRed", label="Test Data LogLoss", linewidth=1.5)

    plt.xlabel("Epochs", fontsize=6.5, fontweight='bold', labelpad=-5)
    plt.ylabel("LogLoss", fontsize=6.5, fontweight='bold', labelpad=-5)

    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=6, fontweight='bold')

    plt.show()


def nscf_auc_curves():
    x = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    train_logloss = np.array([13.9749, 4.3570, 2.7440, 0.8482, 0.7202, 0.6875, 0.6584, 0.6231, 0.6044, 0.5862, 0.5418])
    test_logloss = np.array([15.7243, 5.8326, 3.7541, 1.6620, 0.9174, 0.7735, 0.6913, 0.6287, 0.5824, 0.6307, 0.6744])

    plt.figure(figsize=(6, 3))
    plt.grid(linestyle="--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(x, train_logloss, marker='o', color="SkyBlue", label="Train Data LogLoss", linewidth=1.5)
    plt.plot(x, test_logloss, marker='^', color="IndianRed", label="Test Data LogLoss", linewidth=1.5)

    plt.xlabel("Epochs", fontsize=6.5, fontweight='bold', labelpad=-5)
    plt.ylabel("LogLoss", fontsize=6.5, fontweight='bold', labelpad=-5)

    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=6, fontweight='bold')

    plt.show()


def nscf_auc_logloss():
    aucs = [0.6793, 0.6928, 0.7073, 0.6914, 0.7289, 0.7137, 0.6814, 0.6847, 0.6873]
    logs = [0.6245, 0.6018, 0.5947, 0.6106, 0.5824, 0.5871, 0.6239, 0.6223, 0.6152]
    x = np.arange(len(aucs))
    width = 0.35

    plt.figure(figsize=(6, 3))
    plt.grid(linestyle="--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.bar(x-width/2, aucs, width, color='SkyBlue', label='AUC')
    plt.bar(x+width/2, logs, width, color='IndianRed', label='LogLoss')

    plt.ylabel("AUC and LogLoss")
    plt.title("NS-based CF AUC and LogLoss")
    xlabels = ['K=3,L=128',
               'K=3,L=300',
               'K=3,L=500',
               'K=4,L=128',
               'K=4,L=300',
               'K=4,L=500',
               'K=5,L=128',
               'K=5,L=300',
               'K=5,L=500']
    plt.xticks(x, xlabels, fontsize=6)
    plt.ylim(0.4, 0.9)
    plt.legend()

    plt.show()


def nscf_n_auc_logloss():
    aucs = [0.6545, 0.6879, 0.6933, 0.7051, 0.7289, 0.7172, 0.7335, 0.7231, 0.7368]
    logs = [0.6429, 0.6218, 0.6073, 0.5946, 0.5824, 0.5836, 0.5817, 0.5826, 0.5819]
    x = np.arange(len(aucs))
    width = 0.35

    plt.figure(figsize=(6, 3))
    plt.grid(linestyle="--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.bar(x - width / 2, aucs, width, color='SkyBlue', label='AUC')
    plt.bar(x + width / 2, logs, width, color='IndianRed', label='LogLoss')

    plt.ylabel("AUC and LogLoss")
    plt.title("NS-based CF AUC and LogLoss")
    xlabels = ['N=1*800167',
               'N=2*800167',
               'N=3*800167',
               'N=4*800167',
               'N=5*800167',
               'N=6*800167',
               'N=7*800167',
               'N=8*800167',
               'N=9*800167']
    plt.xticks(x, xlabels, fontsize=6)
    plt.ylim(0.4, 0.9)
    plt.legend()

    plt.show()


def nscf_rmse():
    epochs = [1, 5, 15, 20, 25, 30, 35, 40, 45, 50]
    funksvd_rmses = [1.1462, 1.0471, 0.9826, 0.9445, 0.9326, 0.9218, 0.9102, 0.9017, 0.9014, 0.9013]
    svd_rmses = [1.1227, 1.0328, 0.9681, 0.9228, 0.9183, 0.8974, 0.8827, 0.8824, 0.8820, 0.8819]
    random_nscf_rmses = [1.0236, 1.0130, 0.9425, 0.8944, 0.8743, 0.8728, 0.8794, 0.8771, 0.8767, 0.8757]
    lda_nscf_rmses = [1.0038, 0.9811, 0.9314, 0.8874, 0.8681, 0.8630, 0.8657, 0.8602, 0.8690, 0.8625]
    item2vec_nscf_rmses = [1.0015, 0.9838, 0.9228, 0.8833, 0.8633, 0.8612, 0.8611, 0.8584, 0.8547, 0.8555]
    lfm_nscf_rmses = [0.9017, 0.8936, 0.8765, 0.8691, 0.8516, 0.8488, 0.8444, 0.8430, 0.8459, 0.8465]

    plt.figure(figsize=(6, 3))
    plt.grid(linestyle="--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(epochs, funksvd_rmses, marker='o', color="SkyBlue", label="FunkSVD RMSE", linewidth=1.5)
    plt.plot(epochs, svd_rmses, marker='^', color="IndianRed", label="SVD++ RMSE", linewidth=1.5)
    plt.plot(epochs, random_nscf_rmses, marker='*', color="olive", label="Random NS-based CF RMSE", linewidth=1.5)
    plt.plot(epochs, lda_nscf_rmses, marker='s', color="darkcyan", label="LDA NS-based CF RMSE", linewidth=1.5)
    plt.plot(epochs, item2vec_nscf_rmses, marker='+', color="lightslategray", label="Item2Vec Ns-based CF RMSE", linewidth=1.5)
    plt.plot(epochs, lfm_nscf_rmses, marker="d", color="darkmagenta", label="LFM NS-based CF RMSE", linewidth=1.5)

    plt.xlabel("Epochs", fontsize=6.5, fontweight='bold', labelpad=-5)
    plt.ylabel("RMSE", fontsize=6.5, fontweight='bold', labelpad=0)

    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=6, fontweight='bold')

    plt.show()


def nscf_model_auc_logloss():
    aucs = [0.7073, 0.7143, 0.7289, 0.7317, 0.7304, 0.7378]
    logs = [0.6092, 0.5931, 0.5824, 0.5752, 0.5701, 0.5685]

    x = np.arange(len(aucs))
    width = 0.35

    plt.figure(figsize=(6, 3))
    plt.grid(linestyle="--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.bar(x - width / 2, aucs, width, color='SkyBlue', label='AUC')
    plt.bar(x + width / 2, logs, width, color='IndianRed', label='LogLoss')

    plt.ylabel("AUC and LogLoss")
    plt.title("NS-based CF AUC and LogLoss")
    xlabels = ['FunkSVD',
               'SVD++',
               'Random NS-based CF',
               'LDA NS based CF',
               'Item2Vec NS-based CF',
               'LFM NS-based CF',]
    plt.xticks(x, xlabels, fontsize=5)
    plt.ylim(0.4, 0.9)
    plt.legend()

    plt.show()


def nscf_d_auc_logloss():
    aucs = [0.7207, 0.7289, 0.7324]
    logs = [0.5938, 0.5824, 0.5751]

    x = np.arange(len(aucs))
    width = 0.35

    plt.figure(figsize=(6, 3))
    plt.grid(linestyle="--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.bar(x - width / 2, aucs, width, color='SkyBlue', label='AUC')
    plt.bar(x + width / 2, logs, width, color='IndianRed', label='LogLoss')

    plt.ylabel("AUC and LogLoss")
    plt.title("NS-based CF AUC and LogLoss")
    xlabels = ['ml-100k',
               'ml-1m',
               'ml-10m']
    plt.xticks(x, xlabels, fontsize=8)
    plt.ylim(0.4, 0.9)
    plt.legend()

    plt.show()


def nssd_logloss():
    x = [1, 5, 10, 15, 20, 25, 30]
    train_logloss = [1.53343, 0.58776, 0.02551, 0.02154, 0.01791, 0.01343, 0.01025]
    test_logloss = [1.56235, 0.59634, 0.02593, 0.03697, 0.04703, 0.05191, 0.05879]

    plt.figure(figsize=(6, 3))
    plt.grid(linestyle="--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(x, train_logloss, marker='o', color="SkyBlue", label="Train Data LogLoss", linewidth=1.5)
    plt.plot(x, test_logloss, marker='^', color="IndianRed", label="Test Data LogLoss", linewidth=1.5)

    plt.xlabel("Epochs", fontsize=6.5, fontweight='bold', labelpad=5)
    plt.ylabel("LogLoss", fontsize=6.5, fontweight='bold', labelpad=-5)

    plt.ylim(0.01, 0.6)

    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=6, fontweight='bold')

    plt.show()


def nssd_m_aud_logloss():
    aucs = [0.7981, 0.8132, 0.8246]
    logs = [0.4596, 0.4531, 0.4504]

    x = np.arange(len(aucs))
    width = 0.35

    plt.figure(figsize=(6, 3))
    plt.grid(linestyle="--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.bar(x - width / 2, aucs, width, color='SkyBlue', label='AUC')
    plt.bar(x + width / 2, logs, width, color='IndianRed', label='LogLoss')

    plt.ylabel("AUC and LogLoss")
    plt.title("NS-based Shallow&Deep AUC and LogLoss")
    xlabels = ['Wide&Deep',
               'DeepFM',
               'NS-based Shallow&Deep']
    plt.xticks(x, xlabels, fontsize=8)
    plt.ylim(0.4, 1.1)
    plt.legend()

    plt.show()


def plot_mae():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    users = [10, 20, 30, 40, 50]
    # old_maes = [0.96, 0.92, 0.87, 0.85, 0.84]
    # new_maes = [0.90, 0.85, 0.82, 0.815, 0.817]
    old_maes = [0.89, 0.86, 0.81, 0.795, 0.785]
    new_maes = [0.82, 0.79, 0.77, 0.755, 0.757]

    plt.figure(figsize=(7, 4))
    plt.grid(linestyle="--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(users, old_maes, marker='o', color="SkyBlue", label=u"传统相似用户模型", linewidth=1.5)
    plt.plot(users, new_maes, marker='^', color="IndianRed", label=u"本文相似用户模型", linewidth=1.5)

    plt.xlabel(u"相似近邻数", fontsize=10, fontweight='bold', labelpad=1)
    plt.ylabel("MAE", fontsize=10, fontweight='bold', labelpad=1)

    plt.ylim(0.7, 1)
    plt.xlim(8, 52)

    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=6, fontweight='bold')

    plt.show()


def nscf_epoch_curve():
    x = np.linspace(1, 100, 100)
    train_logloss = [13.9749, ]
    test_logloss = [15.7243, ]

    # plt.figure(figsize=(6, 3))
    # plt.grid(linestyle="--")
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    #
    # plt.plot(x, train_logloss, marker='o', color="SkyBlue", label="Train Data LogLoss", linewidth=1.5)
    # plt.plot(x, test_logloss, marker='^', color="IndianRed", label="Test Data LogLoss", linewidth=1.5)
    #
    # plt.xlabel("Epochs", fontsize=6.5, fontweight='bold', labelpad=-5)
    # plt.ylabel("LogLoss", fontsize=6.5, fontweight='bold', labelpad=-5)
    #
    # plt.legend(loc=0, numpoints=1)
    # leg = plt.gca().get_legend()
    # ltext = leg.get_texts()
    # plt.setp(ltext, fontsize=6, fontweight='bold')
    #
    # plt.show()



def main():
    # nscf_learning_curves()
    # nscf_auc_logloss()
    # nscf_n_auc_logloss()
    # nscf_rmse()
    # nscf_model_auc_logloss()
    # nscf_d_auc_logloss()
    # nssd_logloss()
    # nssd_m_aud_logloss()
    # plot_mae()
    # nscf_epoch_curve()
    nscf_learning_curves()


if __name__ == "__main__":
    main()
    # import random
    # for i in range(20):
    #     print random.randint(0, 9), random.randint(0, 9)




























