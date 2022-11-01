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


def save_hist_model(history, model, evaluation, name):
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "evaluation_accuracy", evaluation[1])
    hist_json_file = f"{name}_history.json"
    with open(hist_json_file, mode="w") as f:
        hist_df.to_json(f)
    eval_json_file = f"{name}_eval_accuracy.json"
    with open(eval_json_file, mode="w") as f:
        {"accuracy": evaluation[1]}.to_json()
        hist_df.to_json(f)
    model.save_pretrained(f"/tmp/{name}_custom_model")


def tensorflow_training_wrapper(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    test_dataset: Dataset,
    saving_name: str,
    num_labels: int = 2,
    epochs: int = 10,
) -> TFAutoModelForSequenceClassification:
    false_shuffle = {
        "columns": ["attention_mask", "input_ids", "token_type_ids"],
        "label_cols": ["labels"],
        "shuffle": False,
        "collate_fn": data_collator,
        "batch_size": 8,
    }
    tf_eval_dataset = eval_dataset.to_tf_dataset(**false_shuffle)

    tf_train_dataset = train_dataset.to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids"],
        label_cols=["labels"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
    )

    tf_test_dataset = test_dataset.to_tf_dataset(**false_shuffle)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=3
    )

    model = TFAutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=num_labels
        )

    model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(),
                # tf.keras.metrics.Precision(),
                # tf.keras.metrics.Recall(),
            ],
        )

    # I just do not understand how keras metrics work with the size of the labels.
    # if num_labels == 2:
    #     model.compile(
    #         optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    #         loss='mse',
    #         # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #         metrics=[
    #             'accuracy',
    #             # tf.keras.metrics.SparseCategoricalAccuracy(),
    #             tf.keras.metrics.Precision(),
    #             tf.keras.metrics.Recall(),
    #         ],
    #     )
    # else:
    #     model.compile(
    #         optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    #         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #         metrics=[
    #             tf.keras.metrics.SparseCategoricalAccuracy(),
    #             # tf.keras.metrics.Precision(),
    #             # tf.keras.metrics.Recall(),
    #         ],
    #     )

        
    history = model.fit(
        tf_train_dataset,
        validation_data=tf_eval_dataset,
        epochs=epochs,
        callbacks=[early_stopping_callback],
    )
    evaluation = model.evaluate(tf_test_dataset)

    save_hist_model(history, model, evaluation, saving_name)

    return history, model, evaluation
