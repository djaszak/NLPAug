import pickle
from datasets import load_dataset, load_metric, concatenate_datasets, ClassLabel
from datasets.fingerprint import Hasher
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import multiprocessing
from NLPAug.character import (
    misspell_data,
    random_switcher_data,
    mid_randomizer_data,
    complete_randomizer_data,
    keyboard_replacer_data,
    remover_data,
    inserter_data,
)

TRAIN = "train"
TEST = "test"

# Variable assignments
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=2
)
metric = load_metric("accuracy")
hasher = Hasher()


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train(training_set, eval_set):
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": training_set,
        "eval_dataset": eval_set,
        "compute_metrics": compute_metrics,
    }
    
    return Trainer(**trainer_kwargs).train()


# def save_results(trainer, name):
#     results = trainer.train()
#     metrics = results.metrics
#     trainer.save_metrics(f"{name}_test", metrics)

imdb_dataset = load_dataset("imdb")
imdb_dataset.pop("unsupervised")
imdb_dataset = imdb_dataset.map(
    tokenize_function, batched=True, num_proc=multiprocessing.cpu_count()
)
imdb_train = imdb_dataset[TRAIN]
imdb_eval = imdb_dataset[TEST]

# Augmented test data
cr_train = imdb_train.select(range(100)).map(
    complete_randomizer_data, num_proc=multiprocessing.cpu_count()
)
# kr_train = imdb_train.map(keyboard_replacer_data, num_proc=multiprocessing.cpu_count())
# mr_train = imdb_train.map(mid_randomizer_data, num_proc=multiprocessing.cpu_count())
# rs_train = imdb_train.map(random_switcher_data, num_proc=multiprocessing.cpu_count())
# inserter_train = imdb_train.map(inserter_data, num_proc=multiprocessing.cpu_count())
# remover_train = imdb_train.map(remover_data, num_proc=multiprocessing.cpu_count())
# misspell_train = imdb_train.map(misspell_data, num_proc=multiprocessing.cpu_count())

# # Extend imdb_train by augmented data
# cr_imdb_train = concatenate_datasets([imdb_train, cr_train])
# kr_imdb_train = concatenate_datasets([imdb_train, kr_train])
# mr_imdb_train = concatenate_datasets([imdb_train, mr_train])
# rs_imdb_train = concatenate_datasets([imdb_train, rs_train])
# inserter_imdb_train = concatenate_datasets([imdb_train, inserter_train])
# remover_imdb_train = concatenate_datasets([imdb_train, remover_train])
# misspell_imdb_train = concatenate_datasets([imdb_train, misspell_train])

# Baseline training
imdb_trained = train(imdb_train, imdb_eval)

# Only augmented datasets
cr_trained = train(imdb_train, imdb_eval)
kr_trained = train(imdb_train, imdb_eval)
mr_trained = train(imdb_train, imdb_eval)
rs_trained = train(imdb_train, imdb_eval)
inserter_trained = train(imdb_train, imdb_eval)
remover_trained = train(imdb_train, imdb_eval)
misspell_trained = train(imdb_train, imdb_eval)

# Baseline data extended by augmented data (50k data instead of 25k)
cr_imdb_trained = train(imdb_train, imdb_eval)
kr_imdb_trained = train(imdb_train, imdb_eval)
mr_imdb_trained = train(imdb_train, imdb_eval)
rs_imdb_trained = train(imdb_train, imdb_eval)
inserter_imdb_trained = train(imdb_train, imdb_eval)
remover_imdb_trained = train(imdb_train, imdb_eval)
misspell_imdb_trained = train(imdb_train, imdb_eval)
# save_results(imdb_trainer, "imdb")
# save_results(cr_trainer, "cr")
# save_results(kr_trainer, "kr")
# save_results(r_trainer, "r")
# save_results(mr_trainer, "mr")
# save_results(remover_trainer, "remover")
# save_results(misspelled_trainer, "misspelled")
# save_results(cr_imdb_eval_trainer, "cr_imdb_eval")
# save_results(kr_imdb_eval_trainer, "kr_imdb_eval")
# save_results(mr_imdb_eval_trainer, "mr_imdb_eval")
# save_results(r_imdb_eval_trainer, "r_imdb_eval")
# save_results(remover_imdb_eval_trainer, "remover_imdb_eval")
# save_results(misspelled_imdb_eval_trainer, "misspelled_imdb_eval")
# save_results(cr_imdb_trainer, "cr_imdb")
# save_results(kr_imdb_trainer, "kr_imdb")
# save_results(mr_imdb_trainer, "mr_imdb")
# save_results(r_imdb_trainer, "r_imdb")
# save_results(remover_imdb_trainer, "remover_imdb")
# save_results(misspelled_imdb_trainer, "misspelled_imdb")
