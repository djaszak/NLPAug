from datasets import load_dataset, load_metric, concatenate_datasets, ClassLabel
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
    # TODO: less epochs
    output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=10
)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=2
)
metric = load_metric("accuracy")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


dataset = load_dataset("imdb")
dataset.pop("unsupervised")
cr_dataset = dataset.map(complete_randomizer_data, num_proc=multiprocessing.cpu_count())

complete_randomizer_imdb_dataset = (
    load_dataset(
        "csv",
        data_files={
            "train": "/home/djaszak/augmented_imdb_datasets/complete_randomizer_imdb_train.csv",
            "test": "/home/djaszak/augmented_imdb_datasets/complete_randomizer_imdb_test.csv",
        },
    )
    .remove_columns("Unnamed: 0")
    .cast_column("label", ClassLabel(num_classes=2, names=["neg", "pos"], id=None))
)
keyboard_replacer_imdb_dataset = (
    load_dataset(
        "csv",
        data_files={
            "train": "/home/djaszak/augmented_imdb_datasets/keyboard_replacer_imdb_train.csv",
            "test": "/home/djaszak/augmented_imdb_datasets/keyboard_replacer_imdb_test.csv",
        },
    )
    .remove_columns("Unnamed: 0")
    .cast_column("label", ClassLabel(num_classes=2, names=["neg", "pos"], id=None))
)
mid_randomizer_imdb_dataset = (
    load_dataset(
        "csv",
        data_files={
            "train": "/home/djaszak/augmented_imdb_datasets/mid_randomizer_imdb_train.csv",
            "test": "/home/djaszak/augmented_imdb_datasets/mid_randomizer_imdb_test.csv",
        },
    )
    .remove_columns("Unnamed: 0")
    .cast_column("label", ClassLabel(num_classes=2, names=["neg", "pos"], id=None))
)
random_switcher_imdb_dataset = (
    load_dataset(
        "csv",
        data_files={
            "train": "/home/djaszak/augmented_imdb_datasets/random_switcher_imdb_train.csv",
            "test": "/home/djaszak/augmented_imdb_datasets/random_switcher_imdb_test.csv",
        },
    )
    .remove_columns("Unnamed: 0")
    .cast_column("label", ClassLabel(num_classes=2, names=["neg", "pos"], id=None))
)
remover_imdb_dataset = (
    load_dataset(
        "csv",
        data_files={
            "train": "/home/djaszak/augmented_imdb_datasets/remover_imdb_train.csv",
            "test": "/home/djaszak/augmented_imdb_datasets/remover_imdb_test.csv",
        },
    )
    .remove_columns("Unnamed: 0")
    .cast_column("label", ClassLabel(num_classes=2, names=["neg", "pos"], id=None))
)
misspelled_imdb_dataset = (
    load_dataset(
        "csv",
        data_files={
            "train": "/home/djaszak/augmented_imdb_datasets/misspelled_imdb_train.csv",
            "test": "/home/djaszak/augmented_imdb_datasets/misspelled_imdb_test.csv",
        },
    )
    .remove_columns("Unnamed: 0")
    .cast_column("label", ClassLabel(num_classes=2, names=["neg", "pos"], id=None))
)


imdb_tokenized_datasets = dataset.map(tokenize_function, batched=True)
imdb_train = imdb_tokenized_datasets[TRAIN]
imdb_eval = imdb_tokenized_datasets[TEST]

cr_tokenized_datasets = complete_randomizer_imdb_dataset.map(
    tokenize_function, batched=True
)
cr_train = cr_tokenized_datasets[TRAIN]
cr_eval = cr_tokenized_datasets[TEST]
cr_imdb_train = concatenate_datasets([imdb_train, cr_train])

kr_tokenized_datasets = keyboard_replacer_imdb_dataset.map(
    tokenize_function, batched=True
)
kr_train = kr_tokenized_datasets[TRAIN]
kr_imdb_train = concatenate_datasets([imdb_train, kr_train])
kr_eval = kr_tokenized_datasets[TEST]

mr_tokenized_datasets = mid_randomizer_imdb_dataset.map(tokenize_function, batched=True)
mr_train = mr_tokenized_datasets[TRAIN]
mr_imdb_train = concatenate_datasets([imdb_train, mr_train])
mr_eval = mr_tokenized_datasets[TEST]

replacer_tokenized_datasets = random_switcher_imdb_dataset.map(
    tokenize_function, batched=True
)
r_train = replacer_tokenized_datasets[TRAIN]
r_imdb_train = concatenate_datasets([imdb_train, r_train])
r_eval = replacer_tokenized_datasets[TEST]

remover_tokenized_datasets = remover_imdb_dataset.map(tokenize_function, batched=True)
remover_train = remover_tokenized_datasets[TRAIN]
remover_imdb_train = concatenate_datasets([imdb_train, remover_train])
remover_eval = remover_tokenized_datasets[TEST]

misspelled_tokenized_datasets = misspelled_imdb_dataset.map(
    tokenize_function, batched=True
)
misspelled_train = misspelled_tokenized_datasets[TRAIN]
misspelled_imdb_train = concatenate_datasets([imdb_train, misspelled_train])
misspelled_eval = misspelled_tokenized_datasets[TEST]

# Use **kwargs
imdb_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=imdb_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)

cr_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=cr_train,
    eval_dataset=cr_eval,
    compute_metrics=compute_metrics,
)

kr_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=kr_train,
    eval_dataset=kr_eval,
    compute_metrics=compute_metrics,
)

mr_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mr_train,
    eval_dataset=mr_eval,
    compute_metrics=compute_metrics,
)

r_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=r_train,
    eval_dataset=r_eval,
    compute_metrics=compute_metrics,
)

remover_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=remover_train,
    eval_dataset=remover_eval,
    compute_metrics=compute_metrics,
)

misspelled_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=misspelled_train,
    eval_dataset=misspelled_eval,
    compute_metrics=compute_metrics,
)

cr_imdb_eval_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=cr_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)

kr_imdb_eval_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=kr_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)

mr_imdb_eval_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mr_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)

r_imdb_eval_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=r_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)

remover_imdb_eval_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=remover_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)

misspelled_imdb_eval_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=misspelled_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)

cr_imdb_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=cr_imdb_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)

kr_imdb_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=kr_imdb_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)

mr_imdb_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mr_imdb_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)

r_imdb_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=r_imdb_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)

remover_imdb_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=remover_imdb_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)

misspelled_imdb_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=misspelled_imdb_train,
    eval_dataset=imdb_eval,
    compute_metrics=compute_metrics,
)


def save_results(trainer, name):
    results = trainer.train()
    metrics = results.metrics
    trainer.save_metrics(f"{name}_test", metrics)


save_results(imdb_trainer, "imdb")
save_results(cr_trainer, "cr")
save_results(kr_trainer, "kr")
save_results(r_trainer, "r")
save_results(mr_trainer, "mr")
save_results(remover_trainer, "remover")
save_results(misspelled_trainer, "misspelled")
save_results(cr_imdb_eval_trainer, "cr_imdb_eval")
save_results(kr_imdb_eval_trainer, "kr_imdb_eval")
save_results(mr_imdb_eval_trainer, "mr_imdb_eval")
save_results(r_imdb_eval_trainer, "r_imdb_eval")
save_results(remover_imdb_eval_trainer, "remover_imdb_eval")
save_results(misspelled_imdb_eval_trainer, "misspelled_imdb_eval")
save_results(cr_imdb_trainer, "cr_imdb")
save_results(kr_imdb_trainer, "kr_imdb")
save_results(mr_imdb_trainer, "mr_imdb")
save_results(r_imdb_trainer, "r_imdb")
save_results(remover_imdb_trainer, "remover_imdb")
save_results(misspelled_imdb_trainer, "misspelled_imdb")
