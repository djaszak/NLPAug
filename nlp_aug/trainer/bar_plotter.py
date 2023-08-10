import os
import math
import json

import matplotlib.pyplot as plt

from pathlib import Path
from nlp_aug import constants
import seaborn as sns
import pandas as pd


def sort_dict_by_value(dict_to_sort):
    return dict(sorted(dict_to_sort.items(), key=lambda item: item[1]))


output_path = (Path(__file__).parent / "output").resolve()
histories = os.listdir(output_path)
accs = {}
val_accs = {}
evaluation_accs = {}


trec_accs = {"size": 5452, "num_labels": 6}
tweet_irony_accs = {"size": 2862, "num_labels": 2}
tweet_climate_accs = {"size": 355, "num_labels": 3}
imdb_accs = {"size": 25000, "num_labels": 2}
rotten_accs = {"size": 8530, "num_labels": 2}

trec_accs = {"size": 5452, "num_labels": 6}
tweet_irony_accs = {"size": 2862, "num_labels": 2}
tweet_climate_accs = {"size": 355, "num_labels": 3}
imdb_accs = {"size": 25000, "num_labels": 2}
rotten_accs = {"size": 8530, "num_labels": 2}

for history in histories:
    current_history = output_path / history
    with current_history.open() as f:
        history_json = json.load(f)
        accs[history] = [
            x for x in history_json["sparse_categorical_accuracy"].values()
        ]
        val_accs[history] = [
            x for x in history_json["val_sparse_categorical_accuracy"].values()
        ]
        evaluation_accs[history] = [
            x for x in history_json["evaluation_accuracy"].values()
        ][0]


for elem in evaluation_accs:
    if elem.startswith(constants.IMDB):
        imdb_accs[elem] = evaluation_accs[elem]
    if elem.startswith(constants.TREC):
        trec_accs[elem] = evaluation_accs[elem]
    if elem.startswith(constants.TWEET_CLIMATE):
        tweet_climate_accs[elem] = evaluation_accs[elem]
    if elem.startswith(constants.TWEET_IRONY):
        tweet_irony_accs[elem] = evaluation_accs[elem]
    if elem.startswith(constants.ROTTEN):
        rotten_accs[elem] = evaluation_accs[elem]

trec_accs = dict(sorted((trec_accs.items())))
tweet_irony_accs = dict(sorted((tweet_irony_accs.items())))
tweet_climate_accs = dict(sorted((tweet_climate_accs.items())))
rotten_accs = dict(sorted((rotten_accs.items())))
imdb_accs = dict(sorted((imdb_accs.items())))


def round_down(n, decimals=0):
    multiplier = 10**decimals
    return math.floor(n * multiplier) / multiplier


def bar_plot_dataset_overall_augmentation(dataset: str, acc_dict: dict) -> None:
    value_list = []
    label_list = []
    diff_list = []
    colors = []

    base_value = 0

    # Format data out of dict into plottable format
    for key, value in acc_dict.items():
        if "0.0" in key:
            value_list.append(value)
            base_value = value
            label_list.append("Base")
            colors.append("b")

    for key, value in acc_dict.items():
        print(key, value)
        if "True" in key:
            key = " ".join(key[: key.find("_True")].split("_"))
        elif "False" in elem[0]:
            key = " ".join(key[: key.find("_False")].split("_"))
        if not "0.0" in key and not "size" in key and not "num_labels" in key:
            value_list.append(value)
            label_list.append(key)
            diff_list.append(base_value - value)
            colors.append("g" if value > base_value else "red")

    # Plot
    sns.set_theme(style="whitegrid")
    # Actual plot + misc stuff

    plt.xlabel("Accuracy", fontsize="8")
    plt.ylabel("Augmentation method", fontsize="8")
    plt.title(dataset)
    plt.barh(label_list, value_list, color=colors)
    plt.xticks(fontsize="6")
    plt.yticks(fontsize="6")
    # plt.ylim(round_down(min(value_list), 1), 1)
    plt.rc("axes", labelsize=3)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=3)  # fontsize of the x tick labels
    plt.rc("ytick", labelsize=3)  # fontsize of the y tick labels
    saving_file_name = dataset + ".png"
    plt.savefig(
        (Path(__file__).parent / "graphs" / saving_file_name).resolve(),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

bar_plot_dataset_overall_augmentation("trec (5452 samples)", trec_accs)
bar_plot_dataset_overall_augmentation("imdb (25000 samples)", imdb_accs)
bar_plot_dataset_overall_augmentation("rotten tomatoes (8530 samples)", rotten_accs)
bar_plot_dataset_overall_augmentation("tweet climate (355 samples)", tweet_climate_accs)
bar_plot_dataset_overall_augmentation("tweet irony (2862 samples)", tweet_irony_accs)
