from transformers import TFAutoModelForSequenceClassification, DefaultDataCollator
from datasets import (
    load_dataset,
    concatenate_datasets,
)
import multiprocessing
from nlp_aug.character.character import augment_hugginface_data
from nlp_aug.trainer.training_utils import (
    tokenize_function,
    tensorflow_training_wrapper,
    save_hist_model,
)
import json

imdb_dataset = load_dataset("imdb")
imdb_dataset.pop("unsupervised")
imdb_dataset = imdb_dataset.map(
    tokenize_function, batched=True, num_proc=multiprocessing.cpu_count()
)
data_collator = DefaultDataCollator(return_tensors="tf")
imdb_test = imdb_dataset["test"]
tf_test_dataset = imdb_test.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)


model_list = [
    "imdb",
    "cr",
    "kr",
    "mr",
    "rs",
    "inserter",
    "remover",
    "misspell",
    "cr_imdb",
    "kr_imdb",
    "mr_imdb",
    "rs_imdb",
    "inserter_imdb",
    "remover_imdb",
    "misspell_imdb",
]

evaluation_accuracy_dict = {}
for name in model_list:
    model = TFAutoModelForSequenceClassification.from_pretrained(
        f"/tmp/{name}_custom_model", num_labels=2
    )
    loss, accuracy = model.evaluate(tf_test_dataset)
    evaluation_accuracy_dict[name] = accuracy

with open('accuracies_from_previous.json', 'w') as fp:
    json.dump(evaluation_accuracy_dict, fp)
