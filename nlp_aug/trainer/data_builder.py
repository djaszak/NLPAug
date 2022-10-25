from datasets import (
    load_dataset,
    concatenate_datasets,
)
import multiprocessing
from nlp_aug.character.character import augment_hugginface_data
from nlp_aug.trainer.training_utils import (
    tokenize_function,
    tensorflow_training_wrapper,
)

TRAIN = "train"
TEST = "test"

imdb_dataset = load_dataset("imdb")
imdb_dataset.pop("unsupervised")
imdb_dataset = imdb_dataset.map(
    tokenize_function, batched=True, num_proc=multiprocessing.cpu_count()
)
split = imdb_dataset[TRAIN].train_test_split(0.8, seed=42)
imdb_train = split[TRAIN]
imdb_eval = split[TEST]
imdb_test = imdb_dataset[TEST]

emotion_dataset = load_dataset("emotion")
emotion_dataset = emotion_dataset.map(
    tokenize_function, batched=True, num_proc=multiprocessing.cpu_count()
)
emotion_train = emotion_dataset["train"]
emotion_eval = emotion_dataset["validation"]
emotion_test = emotion_dataset["test"]


def get_augmentation_fn_kwargs(mode: str):
    return {"augmented_feature": "text", "mode": mode, "augment_probability": 1}


# Augmented test data
cr_train = imdb_train.map(
    augment_hugginface_data,
    num_proc=multiprocessing.cpu_count(),
    fn_kwargs=get_augmentation_fn_kwargs("complete_randomizer"),
)
kr_train = imdb_train.map(
    augment_hugginface_data,
    num_proc=multiprocessing.cpu_count(),
    fn_kwargs=get_augmentation_fn_kwargs("keyboard_replacer"),
)
mr_train = imdb_train.map(
    augment_hugginface_data,
    num_proc=multiprocessing.cpu_count(),
    fn_kwargs=get_augmentation_fn_kwargs("mid_randomizer"),
)
rs_train = imdb_train.map(
    augment_hugginface_data,
    num_proc=multiprocessing.cpu_count(),
    fn_kwargs=get_augmentation_fn_kwargs("random_switcher"),
)
inserter_train = imdb_train.map(
    augment_hugginface_data,
    num_proc=multiprocessing.cpu_count(),
    fn_kwargs=get_augmentation_fn_kwargs("inserter"),
)
remover_train = imdb_train.map(
    augment_hugginface_data,
    num_proc=multiprocessing.cpu_count(),
    fn_kwargs=get_augmentation_fn_kwargs("remover"),
)
misspell_train = imdb_train.map(
    augment_hugginface_data,
    num_proc=multiprocessing.cpu_count(),
    fn_kwargs=get_augmentation_fn_kwargs("misspeller"),
)
# Extend imdb_train by augmented data
cr_imdb_train = concatenate_datasets([imdb_train, cr_train])
kr_imdb_train = concatenate_datasets([imdb_train, kr_train])
mr_imdb_train = concatenate_datasets([imdb_train, mr_train])
rs_imdb_train = concatenate_datasets([imdb_train, rs_train])
inserter_imdb_train = concatenate_datasets([imdb_train, inserter_train])
remover_imdb_train = concatenate_datasets([imdb_train, remover_train])
misspell_imdb_train = concatenate_datasets([imdb_train, misspell_train])