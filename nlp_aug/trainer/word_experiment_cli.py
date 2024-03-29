import multiprocessing

from datasets import concatenate_datasets

from gensim.models import Word2Vec

from nlp_aug import constants
from nlp_aug.word.word import word_augment_huggingface_data
from nlp_aug.trainer.training_utils import tensorflow_training_wrapper
from nlp_aug.trainer.data_loader import load_my_dataset
from nlp_aug.word.word2vec_builder import Word2VecBuilder

import argparse
import datetime
import sys
import time


def run_character_augmentation_experiment(
    dataset: str,
    mode: str = "",
    augment_probability: float = 0.75,
    epochs: int = 5,
    concat: bool = False,
):
    sys.stdout.write("Experiment starts with: \n")
    sys.stdout.write(f"DATASET: {dataset}\n")
    sys.stdout.write(f"MODE: {mode}\n")
    sys.stdout.write(f"AUGMENT_PROBABILITY: {augment_probability}\n")
    sys.stdout.write(f"EPOCHS: {epochs}\n")
    sys.stdout.write(f"CONCAT: {concat}\n")
    sys.stdout.write(f"on time at greenwich meridian: {datetime.datetime.now()}\n")

    time_1 = time.time()
    train_set, test_set, eval_set, num_labels = load_my_dataset(dataset)
    augmented_train = concat_set = None
    if mode:
        model = None
        if mode == constants.EMBEDDING_INSERTER or mode == constants.EMBEDDING_REPLACEMENT:
            sys.stdout.write(f"Starting embedding model creation")
            name = hash(datetime.datetime.now())
            Word2VecBuilder(train_set["text"]).build(f"{name}_word2vec")
            model = Word2Vec.load(f"{name}_word2vec.model")
        augmented_train = train_set.map(
            word_augment_huggingface_data,
            num_proc=multiprocessing.cpu_count(),
            fn_kwargs={
                "augmented_feature": "text",
                "mode": mode,
                "augment_probability": augment_probability,
                "word2vec_model": model
            },
        )
    if concat:
        augmented_train = concatenate_datasets([train_set, augmented_train])

    train_set = augmented_train if augmented_train else train_set
    time_3 = time.time()
    sys.stdout.write(f"Augmentation took about {round((time_3-time_1)/60)} minutes\n")
    tensorflow_training_wrapper(
        train_dataset=train_set,
        eval_dataset=eval_set,
        test_dataset=test_set,
        saving_name=f"{dataset}_{mode}_{augment_probability}_{concat}",
        num_labels=num_labels,
        epochs=epochs,
    )
    time_2 = time.time()
    sys.stdout.write(f"Training took about {round((time_2-time_1)/60)} minutes\n")


parser = argparse.ArgumentParser(
    description="Run character augmentation experiments from the command line"
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    # choices=[
    #     constants.AG_NEWS,
    #     constants.TREC6,
    #     constants.SUBJ,
    #     constants.ROTTEN,
    #     constants.IMDB,
    #     constants.SST2,
    #     constants.COLA,
    # ],
    help="This is the dataset, currently 7 widely known ones are available.",
)
parser.add_argument(
    "--mode",
    type=str,
    default="",
    # choices=[
    #     "complete_randomizer",
    #     "keyboard_replacer",
    #     "mid_randomizer",
    #     "random_switcher",
    #     "inserter",
    #     "remover",
    #     "misspeller",
    # ],
    help="The augmentation mode that should be used for this experiment. If none is provided, nothing is augmented",
)
parser.add_argument(
    "--augment_probability",
    type=float,
    help="The probability with which a token will be augmented, range between 0 and 1.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=5,
    help="The number of epochs used for this experiment.",
)
parser.add_argument(
    "--concat",
    type=bool,
    default=False,
    help="Should the augmented mode be tested or should it be concatenated with the original dataset",
)

print('TROLO')
args = parser.parse_args()
run_character_augmentation_experiment(
    dataset=args.dataset,
    mode=args.mode,
    augment_probability=args.augment_probability,
    epochs=args.epochs,
    concat=args.concat,
)
