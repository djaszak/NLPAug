from datasets import load_dataset, load_metric
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

TRAIN = 'train'
TEST = 'test'

# Variable assignments
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
metric = load_metric("accuracy")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


misspelled_imdb_dataset = load_dataset('csv', data_files={
    'train': '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/misspelled_imdb_train.csv',
    'test': '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/misspelled_imdb_test.csv'
}).remove_columns('Unnamed: 0')

tokenized_datasets = misspelled_imdb_dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets[TRAIN].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets[TEST].shuffle(seed=42).select(range(1000))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
