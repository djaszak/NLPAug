from NLPAug.trainer.training_utils import (
    tensorflow_training_wrapper,
    save_hist_model,
    tokenize_function,
)
from datasets import (
    load_dataset,
)
from NLPAug.character.character import augment_hugginface_data
import multiprocessing

imdb_dataset = load_dataset("imdb")
imdb_dataset.pop("unsupervised")
imdb_dataset = imdb_dataset.map(
    tokenize_function, batched=True, num_proc=multiprocessing.cpu_count()
)
imdb_train = imdb_dataset["train"]
imdb_eval = imdb_dataset["test"]

emotion_dataset = load_dataset("emotion")
emotion_dataset = emotion_dataset.map(
    tokenize_function, batched=True, num_proc=multiprocessing.cpu_count()
)
emotion_train = emotion_dataset["train"]
emotion_eval = emotion_dataset["test"]


def augment_good_techniques(data, augment_probability):
    mr_train = data.map(
        augment_hugginface_data,
        num_proc=multiprocessing.cpu_count(),
        fn_kwargs={
            "augmented_feature": "text",
            "mode": "mid_randomizer",
            "augment_probability": augment_probability,
        },
    )
    inserter_train = data.map(
        augment_hugginface_data,
        num_proc=multiprocessing.cpu_count(),
        fn_kwargs={
            "augmented_feature": "text",
            "mode": "inserter",
            "augment_probability": augment_probability,
        },
    )
    misspell_train = data.map(
        augment_hugginface_data,
        num_proc=multiprocessing.cpu_count(),
        fn_kwargs={
            "augmented_feature": "text",
            "mode": "misspeller",
            "augment_probability": augment_probability,
        },
    )

    return mr_train, inserter_train, misspell_train


# Augmented test data
mr_train_2, inserter_train_2, misspell_train_2 = augment_good_techniques(
    imdb_train, 0.2
)
mr_train_4, inserter_train_4, misspell_train_4 = augment_good_techniques(
    imdb_train, 0.4
)
mr_train_6, inserter_train_6, misspell_train_6 = augment_good_techniques(
    imdb_train, 0.6
)
mr_train_8, inserter_train_8, misspell_train_8 = augment_good_techniques(
    imdb_train, 0.8
)

history, model = tensorflow_training_wrapper(mr_train_2, imdb_eval)
save_hist_model(history, model, "imdb_mr_2")
history, model = tensorflow_training_wrapper(mr_train_4, imdb_eval)
save_hist_model(history, model, "imdb_mr_4")
history, model = tensorflow_training_wrapper(mr_train_6, imdb_eval)
save_hist_model(history, model, "imdb_mr_6")
history, model = tensorflow_training_wrapper(mr_train_8, imdb_eval)
save_hist_model(history, model, "imdb_mr_8")

history, model = tensorflow_training_wrapper(inserter_train_2, imdb_eval)
save_hist_model(history, model, "imdb_inserter_2")
history, model = tensorflow_training_wrapper(inserter_train_4, imdb_eval)
save_hist_model(history, model, "imdb_inserter_4")
history, model = tensorflow_training_wrapper(inserter_train_6, imdb_eval)
save_hist_model(history, model, "imdb_inserter_6")
history, model = tensorflow_training_wrapper(inserter_train_8, imdb_eval)
save_hist_model(history, model, "imdb_inserter_8")

history, model = tensorflow_training_wrapper(misspell_train_2, imdb_eval)
save_hist_model(history, model, "imdb_misspell_2")
history, model = tensorflow_training_wrapper(misspell_train_4, imdb_eval)
save_hist_model(history, model, "imdb_misspell_4")
history, model = tensorflow_training_wrapper(misspell_train_6, imdb_eval)
save_hist_model(history, model, "imdb_misspell_6")
history, model = tensorflow_training_wrapper(misspell_train_8, imdb_eval)
save_hist_model(history, model, "imdb_misspell_8")


# Augmented test data
mr_train_2, inserter_train_2, misspell_train_2 = augment_good_techniques(
    emotion_train, 0.2
)
mr_train_4, inserter_train_4, misspell_train_4 = augment_good_techniques(
    emotion_train, 0.4
)
mr_train_6, inserter_train_6, misspell_train_6 = augment_good_techniques(
    emotion_train, 0.6
)
mr_train_8, inserter_train_8, misspell_train_8 = augment_good_techniques(
    emotion_train, 0.8
)

history, model = tensorflow_training_wrapper(emotion_train, emotion_eval, num_labels=6)
save_hist_model(history, model, "emotion")

history, model = tensorflow_training_wrapper(mr_train_2, emotion_eval, num_labels=6)
save_hist_model(history, model, "emotion_mr_2")
history, model = tensorflow_training_wrapper(mr_train_4, emotion_eval, num_labels=6)
save_hist_model(history, model, "emotion_mr_4")
history, model = tensorflow_training_wrapper(mr_train_6, emotion_eval, num_labels=6)
save_hist_model(history, model, "emotion_mr_6")
history, model = tensorflow_training_wrapper(mr_train_8, emotion_eval, num_labels=6)
save_hist_model(history, model, "emotion_mr_8")

history, model = tensorflow_training_wrapper(
    inserter_train_2, emotion_eval, num_labels=6
)
save_hist_model(history, model, "emotion_inserter_2")
history, model = tensorflow_training_wrapper(
    inserter_train_4, emotion_eval, num_labels=6
)
save_hist_model(history, model, "emotion_inserter_4")
history, model = tensorflow_training_wrapper(
    inserter_train_6, emotion_eval, num_labels=6
)
save_hist_model(history, model, "emotion_inserter_6")
history, model = tensorflow_training_wrapper(
    inserter_train_8, emotion_eval, num_labels=6
)
save_hist_model(history, model, "emotion_inserter_8")

history, model = tensorflow_training_wrapper(
    misspell_train_2, emotion_eval, num_labels=6
)
save_hist_model(history, model, "emotion_misspell_2")
history, model = tensorflow_training_wrapper(
    misspell_train_4, emotion_eval, num_labels=6
)
save_hist_model(history, model, "emotion_misspell_4")
history, model = tensorflow_training_wrapper(
    misspell_train_6, emotion_eval, num_labels=6
)
save_hist_model(history, model, "emotion_misspell_6")
history, model = tensorflow_training_wrapper(
    misspell_train_8, emotion_eval, num_labels=6
)
save_hist_model(history, model, "emotion_misspell_8")