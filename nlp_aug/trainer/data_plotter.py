import json
import os
import pprint

import matplotlib.pyplot as plt
from pathlib import Path
from nlp_aug import constants


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

ag_accs = sort_dict_by_value(ag_accs)
trec6_accs = sort_dict_by_value(trec6_accs)
subj_accs = sort_dict_by_value(subj_accs)
rotten_accs = sort_dict_by_value(rotten_accs)
imdb_accs = sort_dict_by_value(imdb_accs)
sst2_accs = sort_dict_by_value(sst2_accs)
cola_accs = sort_dict_by_value(cola_accs)

pprint.pprint(ag_accs, sort_dicts=False)
pprint.pprint(trec6_accs, sort_dicts=False)
pprint.pprint(subj_accs, sort_dicts=False)
pprint.pprint(rotten_accs, sort_dicts=False)
pprint.pprint(imdb_accs, sort_dicts=False)
pprint.pprint(sst2_accs, sort_dicts=False)
pprint.pprint(cola_accs, sort_dicts=False)

# def plot_things(title, plt):
#     plt.title(title)
#     plt.legend()
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")


# # 5 subplots. 2 on top for accuracy and val_accuracy with only augmented, 2 below with augmented + imdb. imdb on top for itself.
# x = [1, 2, 3]
# plt.subplot(2, 3, 1)
# plt.plot(x, accs["imdb_history.json"], label="imdb_history_acc")
# plt.plot(x, accs["imdb_history.json"], label="imdb_history_val_acc")
# plot_things("IMDB accuracy and validation accuracy", plt)

# plt.subplot(2, 3, 2)

# for history in accs:
#     if "imdb" in history and history != "imdb_history.json":
#         plt.plot(x, accs[history], label=history)
# plot_things("Concatenated datasets accuracies", plt)

# plt.subplot(2, 3, 3)
# for history in accs:
#     if "imdb" not in history:
#         plt.plot(x, accs[history], label=history)
# plot_things("Only augmented dataset accuracies", plt)

# plt.subplot(2, 3, 4)
# for history in val_accs:
#     if "imdb" in history and history != "imdb_history.json":
#         plt.plot(x, val_accs[history], label=history)
# plot_things("Concatenated datasets validation accuracies", plt)

# plt.subplot(2, 3, 5)
# for history in val_accs:
#     if "imdb" not in history:
#         plt.plot(x, val_accs[history], label=history)
# plot_things("Only augmented dataset validation accuracies", plt)

# plt.show()
