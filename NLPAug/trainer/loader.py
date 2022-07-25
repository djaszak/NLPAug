import pandas as pd
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
    TFAutoModelForSequenceClassification,
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


def tensorflow_training_wrapper(
    train_dataset: Dataset, eval_dataset: Dataset
) -> TFAutoModelForSequenceClassification:
    tf_eval_dataset = eval_dataset.to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids"],
        label_cols=["labels"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=8,
    )

    tf_train_dataset = train_dataset.to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids"],
        label_cols=["labels"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
    )

    model = TFAutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.metrics.SparseCategoricalAccuracy(),
            # tf.metrics.Accuracy(),
            # tf.metrics.Precision(),
            # tf.metrics.Recall(),
        ],
    )
    history = model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=3)
    return history, model


def save_hist_model(history, model, name):
    hist_df = pd.DataFrame(history.history)
    hist_json_file = f"{name}_history.json"
    with open(hist_json_file, mode="w") as f:
        hist_df.to_json(f)
    model.save_pretrained(f"/tmp/{name}_custom_model")


imdb_dataset = load_dataset("imdb")
imdb_dataset.pop("unsupervised")
imdb_dataset = imdb_dataset.map(
    tokenize_function, batched=True, num_proc=multiprocessing.cpu_count()
)
imdb_train = imdb_dataset[TRAIN]
imdb_eval = imdb_dataset[TEST]

# Augmented test data
cr_train = imdb_train.map(
    complete_randomizer_data, num_proc=multiprocessing.cpu_count()
)
kr_train = imdb_train.map(keyboard_replacer_data, num_proc=multiprocessing.cpu_count())
mr_train = imdb_train.map(mid_randomizer_data, num_proc=multiprocessing.cpu_count())
rs_train = imdb_train.map(random_switcher_data, num_proc=multiprocessing.cpu_count())
inserter_train = imdb_train.map(inserter_data, num_proc=multiprocessing.cpu_count())
remover_train = imdb_train.map(remover_data, num_proc=multiprocessing.cpu_count())
misspell_train = imdb_train.map(misspell_data, num_proc=multiprocessing.cpu_count())

# Extend imdb_train by augmented data
cr_imdb_train = concatenate_datasets([imdb_train, cr_train])
kr_imdb_train = concatenate_datasets([imdb_train, kr_train])
mr_imdb_train = concatenate_datasets([imdb_train, mr_train])
rs_imdb_train = concatenate_datasets([imdb_train, rs_train])
inserter_imdb_train = concatenate_datasets([imdb_train, inserter_train])
remover_imdb_train = concatenate_datasets([imdb_train, remover_train])
misspell_imdb_train = concatenate_datasets([imdb_train, misspell_train])

# Baseline training
history, model = tensorflow_training_wrapper(imdb_train, imdb_eval)
save_hist_model(history, model, "imdb")

# Augmented training
history, model = tensorflow_training_wrapper(cr_train, imdb_eval)
save_hist_model(history, model, "cr")
history, model = tensorflow_training_wrapper(kr_train, imdb_eval)
save_hist_model(history, model, "kr")
history, model = tensorflow_training_wrapper(mr_train, imdb_eval)
save_hist_model(history, model, "mr")
history, model = tensorflow_training_wrapper(rs_train, imdb_eval)
save_hist_model(history, model, "rs")
history, model = tensorflow_training_wrapper(inserter_train, imdb_eval)
save_hist_model(history, model, "inserter")
history, model = tensorflow_training_wrapper(remover_train, imdb_eval)
save_hist_model(history, model, "remover")
history, model = tensorflow_training_wrapper(misspell_train, imdb_eval)
save_hist_model(history, model, "misspell")

# Augmented and baseline training (50k data instead of 25k)
history, model = tensorflow_training_wrapper(cr_imdb_train, imdb_eval)
save_hist_model(history, model, "cr_imdb")
history, model = tensorflow_training_wrapper(kr_imdb_train, imdb_eval)
save_hist_model(history, model, "kr_imdb")
history, model = tensorflow_training_wrapper(mr_imdb_train, imdb_eval)
save_hist_model(history, model, "mr_imdb")
history, model = tensorflow_training_wrapper(rs_imdb_train, imdb_eval)
save_hist_model(history, model, "rs_imdb")
history, model = tensorflow_training_wrapper(inserter_imdb_train, imdb_eval)
save_hist_model(history, model, "inserter_imdb")
history, model = tensorflow_training_wrapper(remover_imdb_train, imdb_eval)
save_hist_model(history, model, "remover_imdb")
history, model = tensorflow_training_wrapper(misspell_imdb_train, imdb_eval)
save_hist_model(history, model, "misspell_imdb")
