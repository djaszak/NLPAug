import json
import os
import math
import pprint 

from pathlib import Path
from nlp_aug import constants
from tabulate import tabulate

# Color
R = "\033[0;31;40m"  # RED
G = "\033[0;32;40m"  # GREEN
Y = "\033[0;33;40m"  # Yellow
B = "\033[0;34;40m"  # Blue
N = "\033[0m"  # Reset

output_path = (Path(__file__).parent / "output").resolve()
histories = os.listdir(output_path)


def histories_json():
    evaluation_accs = {}
    for history in histories:
        current_history = output_path / history
        with current_history.open() as f:
            history_json = json.load(f)
            evaluation_accs[history] = [
                x for x in history_json["evaluation_accuracy"].values()
            ][0]

    return evaluation_accs


def round_down(n, decimals=0):
    multiplier = 10**decimals
    return math.floor(n * multiplier) / multiplier


trec_accs = {"size": 5452, "num_labels": 6}
tweet_irony_accs = {"size": 2862, "num_labels": 2}
tweet_climate_accs = {"size": 355, "num_labels": 3}
imdb_accs = {"size": 25000, "num_labels": 2}
rotten_accs = {"size": 8530, "num_labels": 2}

accs = dict(sorted((histories_json().items())))
for elem in accs:
    if elem.startswith(constants.TREC):
        trec_accs[elem] = accs[elem]
    if elem.startswith(constants.TWEET_IRONY):
        tweet_irony_accs[elem] = accs[elem]
    if elem.startswith(constants.TWEET_CLIMATE):
        tweet_climate_accs[elem] = accs[elem]
    if elem.startswith(constants.ROTTEN):
        rotten_accs[elem] = accs[elem]
    if elem.startswith(constants.IMDB):
        imdb_accs[elem] = accs[elem]


def transform_dict(accs):
    accs = [[key, val] for key, val in accs.items()]
    base_value = 0
    for elem in accs:
        if "0.0" in elem[0]:
            accs.remove(elem)
            accs.insert(2, elem)
            base_value = elem[1]

    for elem in accs[:2]:
        elem.append(0)

    for elem in accs[2:]:
        elem.append(round_down(elem[1] - base_value, 4))

    for elem in accs:
        if 'True' in elem[0]:
            elem[0] = ' '.join(elem[0][:elem[0].find("_True")].split('_'))
        elif 'False' in elem[0]:
            elem[0] = ' '.join(elem[0][:elem[0].find("_False")].split('_'))
    return accs


accs = [trec_accs, tweet_irony_accs, tweet_climate_accs, rotten_accs, imdb_accs]

formatted_accs = [transform_dict(acc) for acc in accs]

scoring_points = {}

cleaned_accs = []

# print(transform_dict(trec_accs))

for acc in transform_dict(trec_accs):
    if not 'size' in acc or 'num_labels' in acc:
        acc[0] = acc[0].replace('trec ', '')
        cleaned_accs.append(acc)
for acc in transform_dict(tweet_irony_accs):
    if not 'size' in acc or 'num_labels' in acc:
        acc[0] = acc[0].replace('tweet irony ', '')
        cleaned_accs.append(acc)
for acc in transform_dict(tweet_climate_accs):
    if not 'size' in acc or 'num_labels' in acc:
        acc[0] = acc[0].replace('tweet climate ', '')
        cleaned_accs.append(acc)
for acc in transform_dict(imdb_accs):
    if not 'size' in acc or 'num_labels' in acc:
        acc[0] = acc[0].replace('imdb ', '')
        cleaned_accs.append(acc)
for acc in transform_dict(rotten_accs):
    if not 'size' in acc or 'num_labels' in acc:
        acc[0] = acc[0].replace('rotten ', '')
        cleaned_accs.append(acc)


print(cleaned_accs)

for cleaned_acc in cleaned_accs:
    if cleaned_acc[0] not in scoring_points.keys():
        scoring_points[cleaned_acc[0]] = 0
    if cleaned_acc[2] > 0:
        scoring_points[cleaned_acc[0]] += 1
    if cleaned_acc[2] < 0:
        scoring_points[cleaned_acc[0]] -= 1

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(scoring_points)

# print(
#     tabulate(
#         scoring_points,
#         headers=[
#                 "Augmentierungsmethode",
#                 "Punkte",
#         ],
#         tablefmt='latex'
#     )
# )

# for acc in formatted_accs:
#     print(
#         tabulate(
#             acc,
#             headers=[
#                 "Augmentation Name",
#                 "Evaluation Accuracy",
#                 "Accuracy difference compared to base",
#             ],
#             tablefmt="latex",
#         )
#     )


