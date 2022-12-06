import json
import os
import math

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

ag_accs = {"size": 120000, "num_labels": 4}
trec6_accs = {"size": 5452, "num_labels": 6}
subj_accs = {}
rotten_accs = {"size": 8530, "num_labels": 2}
imdb_accs = {"size": 25000, "num_labels": 2}
sst2_accs = {}
cola_accs = {}

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
    if elem.startswith(constants.AG_NEWS):
        ag_accs[elem] = evaluation_accs[elem]
    if elem.startswith(constants.TREC6):
        trec6_accs[elem] = evaluation_accs[elem]
    if elem.startswith(constants.SUBJ):
        subj_accs[elem] = evaluation_accs[elem]
    if elem.startswith(constants.ROTTEN):
        rotten_accs[elem] = evaluation_accs[elem]
    if elem.startswith(constants.IMDB):
        imdb_accs[elem] = evaluation_accs[elem]
    if elem.startswith(constants.SST2):
        sst2_accs[elem] = evaluation_accs[elem]
    if elem.startswith(constants.COLA):
        cola_accs[elem] = evaluation_accs[elem]

ag_accs = dict(sorted((ag_accs.items())))
trec6_accs = dict(sorted((trec6_accs.items())))
subj_accs = dict(sorted((subj_accs.items())))
rotten_accs = dict(sorted((rotten_accs.items())))
imdb_accs = dict(sorted((imdb_accs.items())))
sst2_accs = dict(sorted((sst2_accs.items())))
cola_accs = dict(sorted((cola_accs.items())))


def round_down(n, decimals=0):
    multiplier = 10**decimals
    return math.floor(n * multiplier) / multiplier


def bar_plot_dataset_overall_augmentation(dataset: str, acc_dict: dict) -> None:
    value_list = []
    label_list = []
    diff_list = []
    colors = []
    saving_file_name = dataset + "_0.5_concat"
    base_value = 0
    suffix = '_0.5_True_history.json'

    # Format data out of dict into plottable format
    for key, value in acc_dict.items():
        if "0.0" in key:
            value_list.append(value)
            base_value = value
            label_list.append("Base")
            colors.append("b")

    for key, value in acc_dict.items():
        if not "0.0" in key and not "size" in key and not "num_labels" in key:
            value_list.append(value)
            label_list.append(key[len(dataset) + 1 :][: -len(suffix)])
            diff_list.append(base_value - value)
            colors.append("g" if value > base_value else "red")

    # Format lists into dataframe
    data = {"Augmentation method": label_list, "Accuracy": value_list}
    df = pd.DataFrame(data, columns=["Augmentation method", "Accuracy"])

    # Plot
    plots = sns.barplot(x="Augmentation method", y="Accuracy", data=df)
    # Annotation bars
    for bar in plots.patches:
        plots.annotate(
            f'{format(bar.get_height(), ".4f")} ({round(plots.patches[0].get_height() - bar.get_height(), 4) * -1})',
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="center",
            size=15,
            xytext=(0, 8),
            textcoords="offset points",
        )
    # Actual plot + misc stuff
    plt.xlabel("Augmentation method")
    plt.ylabel("Accuracy")
    plt.title(dataset)
    plt.bar(label_list, value_list, color=colors)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.ylim(round_down(min(value_list), 1), 1)
    plt.show()
    # plt.savefig((Path(__file__).parent / "graphs" / saving_file_name).resolve())

# suffix = '_50_concat'
suffix = ''
bar_plot_dataset_overall_augmentation("trec6"+ suffix, trec6_accs)
# bar_plot_dataset_overall_augmentation("subj"+ suffix, subj_accs)
# bar_plot_dataset_overall_augmentation("rotten"+ suffix, rotten_accs)
# bar_plot_dataset_overall_augmentation("sst2"+ suffix, sst2_accs)
# bar_plot_dataset_overall_augmentation("cola"+ suffix, cola_accs)
