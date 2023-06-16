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
        # works train samples -> 120000 
        raw_dataset = load_dataset("ag_news")
    elif dataset == constants.TREC:
        # works train samples -> 5452 
        raw_dataset = load_dataset("trec")
        raw_dataset = raw_dataset.rename_column("label-coarse", "label")
    elif dataset == constants.ROTTEN:
        # works train samples -> 8530
        raw_dataset = load_dataset("rotten_tomatoes")
    elif dataset == constants.IMDB:
        raw_dataset = load_dataset("imdb")
    elif dataset == constants.SST2:
        raw_dataset = load_dataset("sst2")
    elif dataset == constants.COLA:
        raw_dataset = load_dataset("linxinyuan/cola")
    elif dataset == constants.TWEET_IRONY:
        raw_dataset = load_dataset("tweet_eval", "irony")
    elif dataset == constants.TWEET_CLIMATE:
        raw_dataset = load_dataset("tweet_eval", "stance_climate")
    else:
        print("dataset not known")
        exit(-1)

    raw_dataset = raw_dataset.map(
        tokenize_function, batched=True
    )
    split = raw_dataset[TRAIN].train_test_split(0.8, seed=42)

    print('TESTING WITH ONLY 10 SAMPLES!!!!')
    return (
        split[TRAIN].select(range(10)),
        split[TEST].select(range(10)),
        raw_dataset[TEST].select(range(10)),
        np.unique(raw_dataset["train"]["label"]).shape[0],
    )
