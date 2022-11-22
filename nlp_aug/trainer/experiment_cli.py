import multiprocessing

from datasets import concatenate_datasets

from nlp_aug import constants
from nlp_aug.character.character import augment_huggingface_data
from nlp_aug.trainer.training_utils import tensorflow_training_wrapper
from nlp_aug.trainer.data_loader import load_my_dataset

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
        augmented_train = train_set.map(
            augment_huggingface_data,
            num_proc=multiprocessing.cpu_count(),
            fn_kwargs={
                "augmented_feature": "text",
                "mode": mode,
                "augment_probability": augment_probability,
            },
        )
    if concat:
        train_set = concatenate_datasets([train_set, augmented_train])

    train_set = augmented_train if augmented_train else train_set

    tensorflow_training_wrapper(
        train_dataset=train_set,
        eval_dataset=eval_set,
        test_dataset=test_set,
        saving_name=f"{dataset}_{mode}_{augment_probability}_{concat}",
        num_labels=num_labels,
        epochs=epochs,
    )
    time_2 = time.time()
    sys.stdout.write(f"Augmentation took about {round((time_2-time_1)/60)} minutes\n")


parser = argparse.ArgumentParser(
    description="Run character augmentation experiments from the command line"
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=[
        constants.AG_NEWS,
        constants.TREC6,
        constants.SUBJ,
        constants.ROTTEN,
        constants.IMDB,
        constants.SST2,
        constants.COLA,
    ],
    help="This is the dataset, currently 7 widely known ones are available.",
)
parser.add_argument(
    "--mode",
    type=str,
    default="",
    choices=[
        "complete_randomizer",
        "keyboard_replacer",
        "mid_randomizer",
        "random_switcher",
        "inserter",
        "remover",
        "misspeller",
    ],
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

args = parser.parse_args()

run_character_augmentation_experiment(
    dataset=args.dataset,
    mode=args.mode,
    augment_probability=args.augment_probability,
    epochs=args.epochs,
    concat=args.concat,
)

# def get_augmentation_fn_kwargs(mode: str):
#     return {"augmented_feature": "text", "mode": mode, "augment_probability": 0.75}

# def basic_character_pipeline(dataset: str):
#     train_set, test_set, eval_set, num_labels = load_my_dataset(dataset)

#     # Augmented test data
#     time1 = time.time()
#     cr_train = train_set.map(
#         augment_huggingface_data,
#         num_proc=multiprocessing.cpu_count(),
#         fn_kwargs=get_augmentation_fn_kwargs("complete_randomizer"),
#     )
#     kr_train = train_set.map(
#         augment_huggingface_data,
#         num_proc=multiprocessing.cpu_count(),
#         fn_kwargs=get_augmentation_fn_kwargs("keyboard_replacer"),
#     )
#     mr_train = train_set.map(
#         augment_huggingface_data,
#         num_proc=multiprocessing.cpu_count(),
#         fn_kwargs=get_augmentation_fn_kwargs("mid_randomizer"),
#     )
#     rs_train = train_set.map(
#         augment_huggingface_data,
#         num_proc=multiprocessing.cpu_count(),
#         fn_kwargs=get_augmentation_fn_kwargs("random_switcher"),
#     )
#     inserter_train = train_set.map(
#         augment_huggingface_data,
#         num_proc=multiprocessing.cpu_count(),
#         fn_kwargs=get_augmentation_fn_kwargs("inserter"),
#     )
#     remover_train = train_set.map(
#         augment_huggingface_data,
#         num_proc=multiprocessing.cpu_count(),
#         fn_kwargs=get_augmentation_fn_kwargs("remover"),
#     )
#     misspell_train = train_set.map(
#         augment_huggingface_data,
#         num_proc=multiprocessing.cpu_count(),
#         fn_kwargs=get_augmentation_fn_kwargs("misspeller"),
#     )
#     # Extend train_set by augmented data
#     cr_concat_set = concatenate_datasets([train_set, cr_train])
#     kr_concat_set = concatenate_datasets([train_set, kr_train])
#     mr_concat_set = concatenate_datasets([train_set, mr_train])
#     rs_concat_set = concatenate_datasets([train_set, rs_train])
#     inserter_concat_set = concatenate_datasets([train_set, inserter_train])
#     remover_concat_set = concatenate_datasets([train_set, remover_train])
#     misspell_concat_set = concatenate_datasets([train_set, misspell_train])
#     time2 = time.time()
#     print(f'Augmentation took ({(time2-time1)/60})) m')
#     print('All data augmented')

# tensorflow_training_wrapper(train_set, eval_set, test_set, dataset, num_labels)
# time1 = time.time()
# tensorflow_training_wrapper(
#     cr_train, eval_set, test_set, dataset + f"{dataset}_cr", num_labels
# )
# time2 = time.time()
# print(f'Normal training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         kr_train, eval_set, test_set, f"{dataset}_kr", num_labels
#     )
#     time2 = time.time()
#     print(f'KR augmented training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         mr_train, eval_set, test_set, f"{dataset}_mr", num_labels
#     )
#     time2 = time.time()
#     print(f'MR augmented training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         rs_train, eval_set, test_set, f"{dataset}_rs", num_labels
#     )
#     time2 = time.time()
#     print(f'RS augmented training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         inserter_train, eval_set, test_set, f"{dataset}_inserter", num_labels
#     )
#     time2 = time.time()
#     print(f'Inserter augmented training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         remover_train, eval_set, test_set, f"{dataset}_remover", num_labels
#     )
#     time2 = time.time()
#     print(f'Remover augmented training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         misspell_train, eval_set, test_set, f"{dataset}_misspell", num_labels
#     )
#     time2 = time.time()
#     print(f'Misspell augmented training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         cr_concat_set, eval_set, test_set, f"{dataset}_cr_concat", num_labels
#     )
#     time2 = time.time()
#     print(f'CR augmented training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         kr_concat_set, eval_set, test_set, f"{dataset}_kr_concat", num_labels
#     )
#     time2 = time.time()
#     print(f'KR augmented training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         mr_concat_set, eval_set, test_set, f"{dataset}_mr_concat", num_labels
#     )
#     time2 = time.time()
#     print(f'MR augmented training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         rs_concat_set, eval_set, test_set, f"{dataset}_rs_concat", num_labels
#     )
#     time2 = time.time()
#     print(f'RS Concat augmented training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         inserter_concat_set,
#         eval_set,
#         test_set,
#         f"{dataset}_inserter_concat",
#         num_labels,
#     )
#     time2 = time.time()
#     print(f'inserter Concat augmented training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         remover_concat_set, eval_set, test_set, f"{dataset}_remover_concat", num_labels
#     )
#     time2 = time.time()
#     print(f'remover Concat augmented training took ({(time2-time1)/60})) m')

#     time1 = time.time()
#     tensorflow_training_wrapper(
#         misspell_concat_set,
#         eval_set,
#         test_set,
#         f"{dataset}_misspell_concat",
#         num_labels,
#     )
#     time2 = time.time()
#     print(f'misspell Concat augmented training took ({(time2-time1)/60})) m')


# def promising_character_techniques_pipeline(dataset: str):
#     train_set, test_set, eval_set, num_labels = load_my_dataset(dataset)

#     def _augment_good_techniques(data, augment_probability):
#         mr_train = data.map(
#             augment_huggingface_data,
#             num_proc=multiprocessing.cpu_count(),
#             fn_kwargs={
#                 "augmented_feature": "text",
#                 "mode": "mid_randomizer",
#                 "augment_probability": augment_probability,
#             },
#         )
#         inserter_train = data.map(
#             augment_huggingface_data,
#             num_proc=multiprocessing.cpu_count(),
#             fn_kwargs={
#                 "augmented_feature": "text",
#                 "mode": "inserter",
#                 "augment_probability": augment_probability,
#             },
#         )
#         misspell_train = data.map(
#             augment_huggingface_data,
#             num_proc=multiprocessing.cpu_count(),
#             fn_kwargs={
#                 "augmented_feature": "text",
#                 "mode": "misspeller",
#                 "augment_probability": augment_probability,
#             },
#         )

#         return mr_train, inserter_train, misspell_train

#     mr_train_2, inserter_train_2, misspell_train_2 = _augment_good_techniques(
#         train_set, 0.2
#     )
#     mr_train_4, inserter_train_4, misspell_train_4 = _augment_good_techniques(
#         train_set, 0.4
#     )
#     mr_train_6, inserter_train_6, misspell_train_6 = _augment_good_techniques(
#         train_set, 0.6
#     )
#     mr_train_8, inserter_train_8, misspell_train_8 = _augment_good_techniques(
#         train_set, 0.8
#     )
#     tensorflow_training_wrapper(mr_train_2, eval_set, test_set, f"{dataset}_mr_2")
#     tensorflow_training_wrapper(mr_train_4, eval_set, test_set, f"{dataset}_mr_4")
#     tensorflow_training_wrapper(mr_train_6, eval_set, test_set, f"{dataset}_mr_6")
#     tensorflow_training_wrapper(mr_train_8, eval_set, test_set, f"{dataset}_mr_8")

#     tensorflow_training_wrapper(mr_train_2, eval_set, test_set, f"{dataset}_mr_2")
#     tensorflow_training_wrapper(mr_train_4, eval_set, test_set, f"{dataset}_mr_4")
#     tensorflow_training_wrapper(mr_train_6, eval_set, test_set, f"{dataset}_mr_6")
#     tensorflow_training_wrapper(mr_train_8, eval_set, test_set, f"{dataset}_mr_8")

#     tensorflow_training_wrapper(
#         inserter_train_2, eval_set, test_set, f"{dataset}_inserter_2"
#     )
#     tensorflow_training_wrapper(
#         inserter_train_4, eval_set, test_set, f"{dataset}_inserter_4"
#     )
#     tensorflow_training_wrapper(
#         inserter_train_6, eval_set, test_set, f"{dataset}_inserter_6"
#     )
#     tensorflow_training_wrapper(
#         inserter_train_8, eval_set, test_set, f"{dataset}_inserter_8"
#     )

#     tensorflow_training_wrapper(
#         misspell_train_2, eval_set, test_set, f"{dataset}_misspell_2"
#     )
#     tensorflow_training_wrapper(
#         misspell_train_4, eval_set, test_set, f"{dataset}_misspell_2"
#     )
#     tensorflow_training_wrapper(
#         misspell_train_6, eval_set, test_set, f"{dataset}_misspell_2"
#     )
#     tensorflow_training_wrapper(
#         misspell_train_8, eval_set, test_set, f"{dataset}_misspell_2"
#     )
