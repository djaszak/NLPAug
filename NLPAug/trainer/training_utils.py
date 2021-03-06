import pandas as pd
from datasets import (
    Dataset,
    load_metric,
)
from datasets.fingerprint import Hasher
import numpy as np
import tensorflow as tf
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DefaultDataCollator,
    TrainingArguments,
    TFAutoModelForSequenceClassification,
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
    train_dataset: Dataset, eval_dataset: Dataset, num_labels: int = 2
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
        "bert-base-cased", num_labels=num_labels
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
