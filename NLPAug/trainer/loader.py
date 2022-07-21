import pickle
from datasets import (
    Dataset,
    load_dataset,
    load_metric,
    concatenate_datasets,
    ClassLabel,
)
from datasets.fingerprint import Hasher
import numpy as np
import tensorflow as tf
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
    TFAutoModelForSequenceClassification
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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

TRAIN = "train"
TEST = "test"

# Variable assignments
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="steps",
    num_train_epochs=1,
    load_best_model_at_end=True,
)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=2
)
metric = load_metric("accuracy")
hasher = Hasher()
data_collator = DefaultDataCollator(return_tensors="tf")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train(training_set: Dataset):
    splitted = training_set.train_test_split(test_size=0.2, seed=42)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": splitted["train"],
        "eval_dataset": splitted["test"],
        "compute_metrics": compute_metrics,
    }

    return Trainer(**trainer_kwargs).train()


def save_results(trainer: Trainer, name):
    results = trainer.train()
    metrics = results.metrics
    trainer.save_metrics(f"{name}_test", metrics)


imdb_dataset = load_dataset("imdb")
imdb_dataset.pop("unsupervised")
imdb_dataset = imdb_dataset.map(
    tokenize_function, batched=True, num_proc=multiprocessing.cpu_count()
)
imdb_train = imdb_dataset[TRAIN]
imdb_eval = imdb_dataset[TEST]

# # Augmented test data
cr_train = imdb_train.select(range(5000)).map(
    complete_randomizer_data, num_proc=multiprocessing.cpu_count()
)
# cr_trained = train(cr_train)
# print("This is the evaluation that should be printed", cr_trained)

# Load trained model
# model_path = "test_trainer/checkpoint-500"
# model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# # Define test trainer
# test_trainer = Trainer(model)
# print(test_trainer.predict(imdb_eval.select(range(1000))))

tf_train_dataset = cr_train.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

tf_validation_dataset = imdb_eval.select(range(500)).to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)

model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)

history = model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)

print(history)

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
# imdb_trained = train(imdb_train, imdb_eval)

# # Only augmented datasets
# cr_trained = train(imdb_train, imdb_eval)
# kr_trained = train(imdb_train, imdb_eval)
# mr_trained = train(imdb_train, imdb_eval)
# rs_trained = train(imdb_train, imdb_eval)
# inserter_trained = train(imdb_train, imdb_eval)
# remover_trained = train(imdb_train, imdb_eval)
# misspell_trained = train(imdb_train, imdb_eval)

# # Baseline data extended by augmented data (50k data instead of 25k)
# cr_imdb_trained = train(imdb_train, imdb_eval)
# kr_imdb_trained = train(imdb_train, imdb_eval)
# mr_imdb_trained = train(imdb_train, imdb_eval)
# rs_imdb_trained = train(imdb_train, imdb_eval)
# inserter_imdb_trained = train(imdb_train, imdb_eval)
# remover_imdb_trained = train(imdb_train, imdb_eval)
# misspell_imdb_trained = train(imdb_train, imdb_eval)
