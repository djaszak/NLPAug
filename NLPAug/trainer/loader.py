from datasets import (
    load_dataset,
    concatenate_datasets,
)
import multiprocessing
from NLPAug.character.character import augment_hugginface_data
from NLPAug.trainer.training_utils import (
    tokenize_function,
    tensorflow_training_wrapper,
    save_hist_model,
)

TRAIN = "train"
TEST = "test"

imdb_dataset = load_dataset("imdb")
imdb_dataset.pop("unsupervised")
imdb_dataset = imdb_dataset.map(
    tokenize_function, batched=True, num_proc=multiprocessing.cpu_count()
)
split = imdb_dataset[TRAIN].train_test_split(0.8)
imdb_train = split[TRAIN].select(range(1000))
imdb_eval = split[TEST].select(range(1000))
imdb_test = imdb_dataset[TEST].select(range(1000))


def get_augmentation_fn_kwargs(mode: str):
    return {"augmented_feature": "text", "mode": mode, "augment_probability": 1}


# Augmented test data
cr_train = imdb_train.map(
    augment_hugginface_data,
    num_proc=multiprocessing.cpu_count(),
    fn_kwargs=get_augmentation_fn_kwargs("complete_randomizer"),
)
# kr_train = imdb_train.map(
#     augment_hugginface_data,
#     num_proc=multiprocessing.cpu_count(),
#     fn_kwargs=get_augmentation_fn_kwargs("keyboard_replacer"),
# )
# mr_train = imdb_train.map(
#     augment_hugginface_data,
#     num_proc=multiprocessing.cpu_count(),
#     fn_kwargs=get_augmentation_fn_kwargs("mid_randomizer"),
# )
# rs_train = imdb_train.map(
#     augment_hugginface_data,
#     num_proc=multiprocessing.cpu_count(),
#     fn_kwargs=get_augmentation_fn_kwargs("random_switcher"),
# )
# inserter_train = imdb_train.map(
#     augment_hugginface_data,
#     num_proc=multiprocessing.cpu_count(),
#     fn_kwargs=get_augmentation_fn_kwargs("inserter"),
# )
# remover_train = imdb_train.map(
#     augment_hugginface_data,
#     num_proc=multiprocessing.cpu_count(),
#     fn_kwargs=get_augmentation_fn_kwargs("remover"),
# )
# misspell_train = imdb_train.map(
#     augment_hugginface_data,
#     num_proc=multiprocessing.cpu_count(),
#     fn_kwargs=get_augmentation_fn_kwargs("misspeller"),
# )
# # Extend imdb_train by augmented data
# cr_imdb_train = concatenate_datasets([imdb_train, cr_train])
# kr_imdb_train = concatenate_datasets([imdb_train, kr_train])
# mr_imdb_train = concatenate_datasets([imdb_train, mr_train])
# rs_imdb_train = concatenate_datasets([imdb_train, rs_train])
# inserter_imdb_train = concatenate_datasets([imdb_train, inserter_train])
# remover_imdb_train = concatenate_datasets([imdb_train, remover_train])
# misspell_imdb_train = concatenate_datasets([imdb_train, misspell_train])

eval_dict = {}

# # Baseline training
# history, model, evaluation = tensorflow_training_wrapper(imdb_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "imdb")
# # Augmented training
history, model, evaluation = tensorflow_training_wrapper(cr_train, imdb_eval, imdb_test)
save_hist_model(history, model, "cr")
# history, model, evaluation = tensorflow_training_wrapper(kr_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "kr")
# history, model, evaluation = tensorflow_training_wrapper(mr_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "mr")
# history, model, evaluation = tensorflow_training_wrapper(rs_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "rs")
# history, model, evaluation = tensorflow_training_wrapper(inserter_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "inserter")
# history, model, evaluation = tensorflow_training_wrapper(remover_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "remover")
# history, model, evaluation = tensorflow_training_wrapper(misspell_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "misspell")

# # Augmented and baseline training (50k data instead of 25k)
# history, model, evaluation = tensorflow_training_wrapper(cr_imdb_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "cr_imdb")
# history, model, evaluation = tensorflow_training_wrapper(kr_imdb_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "kr_imdb")
# history, model, evaluation = tensorflow_training_wrapper(mr_imdb_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "mr_imdb")
# history, model, evaluation = tensorflow_training_wrapper(rs_imdb_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "rs_imdb")
# history, model, evaluation = tensorflow_training_wrapper(inserter_imdb_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "inserter_imdb")
# history, model, evaluation = tensorflow_training_wrapper(remover_imdb_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "remover_imdb")
# history, model, evaluation = tensorflow_training_wrapper(misspell_imdb_train, imdb_eval, imdb_test)
# save_hist_model(history, model, "misspell_imdb")
