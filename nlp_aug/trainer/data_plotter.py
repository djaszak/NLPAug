import json
import os

import matplotlib.pyplot as plt
from pathlib import Path

history_path = Path(os.getcwd()) / "training_history"
histories = os.listdir(history_path)

accs = {}
val_accs = {}

x = [1, 2, 3]

for history in histories:
    current_history = history_path / history
    with current_history.open() as f:
        history_json = json.load(f)
        accs[history] = [
            x for x in history_json["sparse_categorical_accuracy"].values()
        ]
        val_accs[history] = [
            x for x in history_json["val_sparse_categorical_accuracy"].values()
        ]


def plot_things(title, plt):
    plt.title(title)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")


# 5 subplots. 2 on top for accuracy and val_accuracy with only augmented, 2 below with augmented + imdb. imdb on top for itself.
plt.subplot(2, 3, 1)
plt.plot(x, accs["imdb_history.json"], label="imdb_history_acc")
plt.plot(x, accs["imdb_history.json"], label="imdb_history_val_acc")
plot_things("IMDB accuracy and validation accuracy", plt)

plt.subplot(2, 3, 2)

for history in accs:
    if "imdb" in history and history != "imdb_history.json":
        plt.plot(x, accs[history], label=history)
plot_things("Concatenated datasets accuracies", plt)

plt.subplot(2, 3, 3)
for history in accs:
    if "imdb" not in history:
        plt.plot(x, accs[history], label=history)
plot_things("Only augmented dataset accuracies", plt)

plt.subplot(2, 3, 4)
for history in val_accs:
    if "imdb" in history and history != "imdb_history.json":
        plt.plot(x, val_accs[history], label=history)
plot_things("Concatenated datasets validation accuracies", plt)

plt.subplot(2, 3, 5)
for history in val_accs:
    if "imdb" not in history:
        plt.plot(x, val_accs[history], label=history)
plot_things("Only augmented dataset validation accuracies", plt)

plt.show()
