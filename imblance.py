#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""imblance"""


import os


def imblance(path, sep):
    """
    imblance
    Args:
        path: file path
        sep: separator
    """
    if not os.path.exists(path):
        return {}
    user_action_numbers = {}
    with open(path, "rb") as fp:
        for line in fp:
            userid, itemid, rating, timestamp = line.strip().split(sep)
            if userid not in user_action_numbers:
                user_action_numbers[userid] = 1
            else:
                user_action_numbers[userid] += 1

    user_action = sorted(user_action_numbers.items(), key=lambda x: x[1], reverse=True)

    return [x[1] for x in user_action]


def plot(array):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    x = np.array(range(len(array)))
    y = np.array(array)

    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")
    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(x, y, marker="o", color="SkyBlue", label="ml-10m imbalance", linewidth=1.5)

    plt.xlabel("User", fontsize=16, fontweight="bold")
    plt.ylabel("User Action Numbers", fontsize=16, fontweight="bold")

    plt.legend(loc=0, numpoints=1)
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=16, fontweight='bold')

    plt.savefig("ml-10m-imblance.png")


def main():
    path = "dataset/ml-10m/ratings.dat"
    plot(imblance(path, "::"))



if __name__ == "__main__":
    main()
