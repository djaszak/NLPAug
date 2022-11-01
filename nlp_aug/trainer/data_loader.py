from asyncio import constants
from datasets import (
    load_dataset,
    concatenate_datasets,
)
import numpy as np
import multiprocessing
from nlp_aug import constants
from nlp_aug.character.character import augment_huggingface_data
from nlp_aug.trainer.training_utils import (
    tokenize_function,
)

TRAIN = "train"
TEST = "test"


def load_my_dataset(dataset: str):
    if dataset == constants.AG_NEWS:
        # works
        raw_dataset = load_dataset("ag_news")
    elif dataset == constants.TREC6:
        # works
        raw_dataset = load_dataset("trec")
        raw_dataset = raw_dataset.rename_column("label-coarse", "label")
    elif dataset == constants.SUBJ:
        # works
        raw_dataset = load_dataset("SetFit/subj")
    elif dataset == constants.ROTTEN:
        # works
        raw_dataset = load_dataset("rotten_tomatoes")
    elif dataset == constants.IMDB:
        # works
        raw_dataset = load_dataset("imdb")
    elif dataset == constants.SST2:
        raw_dataset = load_dataset("gpt3mix/sst2")
        #  raw_dataset = raw_dataset.rename_column("sentence", "text")
    elif dataset == constants.COLA:
        raw_dataset = load_dataset("linxinyuan/cola")
        # raw_dataset = raw_dataset.rename_column("sentence", "text")
    else:
        print("dataset not known")
        exit(-1)

    raw_dataset = raw_dataset.map(
        tokenize_function, batched=True, num_proc=multiprocessing.cpu_count()
    )

    split = raw_dataset[TRAIN].train_test_split(0.8, seed=42)

    return (
        split[TRAIN],
        split[TEST],
        raw_dataset[TEST],
        np.unique(raw_dataset["train"]["label"]).shape[0],
    )
