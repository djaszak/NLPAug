import multiprocessing

from datasets import concatenate_datasets

from nlp_aug.character.character import augment_huggingface_data
from nlp_aug.trainer.training_utils import tensorflow_training_wrapper
from nlp_aug.trainer.data_loader import load_dataset


def get_augmentation_fn_kwargs(mode: str):
    return {"augmented_feature": "text", "mode": mode, "augment_probability": 1}


def basic_character_pipeline(dataset: str):
    train_set, test_set, eval_set, num_labels = load_dataset(dataset)

    # Augmented test data
    cr_train = train_set.map(
        augment_huggingface_data,
        num_proc=multiprocessing.cpu_count(),
        fn_kwargs=get_augmentation_fn_kwargs("complete_randomizer"),
    )
    kr_train = train_set.map(
        augment_huggingface_data,
        num_proc=multiprocessing.cpu_count(),
        fn_kwargs=get_augmentation_fn_kwargs("keyboard_replacer"),
    )
    mr_train = train_set.map(
        augment_huggingface_data,
        num_proc=multiprocessing.cpu_count(),
        fn_kwargs=get_augmentation_fn_kwargs("mid_randomizer"),
    )
    rs_train = train_set.map(
        augment_huggingface_data,
        num_proc=multiprocessing.cpu_count(),
        fn_kwargs=get_augmentation_fn_kwargs("random_switcher"),
    )
    inserter_train = train_set.map(
        augment_huggingface_data,
        num_proc=multiprocessing.cpu_count(),
        fn_kwargs=get_augmentation_fn_kwargs("inserter"),
    )
    remover_train = train_set.map(
        augment_huggingface_data,
        num_proc=multiprocessing.cpu_count(),
        fn_kwargs=get_augmentation_fn_kwargs("remover"),
    )
    misspell_train = train_set.map(
        augment_huggingface_data,
        num_proc=multiprocessing.cpu_count(),
        fn_kwargs=get_augmentation_fn_kwargs("misspeller"),
    )
    # Extend train_set by augmented data
    cr_concat_set = concatenate_datasets([train_set, cr_train])
    kr_concat_set = concatenate_datasets([train_set, kr_train])
    mr_concat_set = concatenate_datasets([train_set, mr_train])
    rs_concat_set = concatenate_datasets([train_set, rs_train])
    inserter_concat_set = concatenate_datasets([train_set, inserter_train])
    remover_concat_set = concatenate_datasets([train_set, remover_train])
    misspell_concat_set = concatenate_datasets([train_set, misspell_train])

    tensorflow_training_wrapper(train_set, eval_set, test_set, dataset, num_labels)

    tensorflow_training_wrapper(
        cr_train, eval_set, test_set, dataset + f"{dataset}_cr", num_labels
    )
    tensorflow_training_wrapper(
        kr_train, eval_set, test_set, f"{dataset}_kr", num_labels
    )
    tensorflow_training_wrapper(
        mr_train, eval_set, test_set, f"{dataset}_mr", num_labels
    )
    tensorflow_training_wrapper(
        rs_train, eval_set, test_set, f"{dataset}_rs", num_labels
    )
    tensorflow_training_wrapper(
        inserter_train, eval_set, test_set, f"{dataset}_inserter", num_labels
    )
    tensorflow_training_wrapper(
        remover_train, eval_set, test_set, f"{dataset}_remover", num_labels
    )
    tensorflow_training_wrapper(
        misspell_train, eval_set, test_set, f"{dataset}_misspell", num_labels
    )

    tensorflow_training_wrapper(
        cr_concat_set, eval_set, test_set, f"{dataset}_cr_concat", num_labels
    )
    tensorflow_training_wrapper(
        kr_concat_set, eval_set, test_set, f"{dataset}_kr_concat", num_labels
    )
    tensorflow_training_wrapper(
        mr_concat_set, eval_set, test_set, f"{dataset}_mr_concat", num_labels
    )
    tensorflow_training_wrapper(
        rs_concat_set, eval_set, test_set, f"{dataset}_rs_concat", num_labels
    )
    tensorflow_training_wrapper(
        inserter_concat_set,
        eval_set,
        test_set,
        f"{dataset}_inserter_concat",
        num_labels,
    )
    tensorflow_training_wrapper(
        remover_concat_set, eval_set, test_set, f"{dataset}_remover_concat", num_labels
    )
    tensorflow_training_wrapper(
        misspell_concat_set,
        eval_set,
        test_set,
        f"{dataset}_misspell_concat",
        num_labels,
    )


def promising_character_techniques_pipeline(dataset: str):
    train_set, test_set, eval_set, num_labels = load_dataset(dataset)

    def _augment_good_techniques(data, augment_probability):
        mr_train = data.map(
            augment_huggingface_data,
            num_proc=multiprocessing.cpu_count(),
            fn_kwargs={
                "augmented_feature": "text",
                "mode": "mid_randomizer",
                "augment_probability": augment_probability,
            },
        )
        inserter_train = data.map(
            augment_huggingface_data,
            num_proc=multiprocessing.cpu_count(),
            fn_kwargs={
                "augmented_feature": "text",
                "mode": "inserter",
                "augment_probability": augment_probability,
            },
        )
        misspell_train = data.map(
            augment_huggingface_data,
            num_proc=multiprocessing.cpu_count(),
            fn_kwargs={
                "augmented_feature": "text",
                "mode": "misspeller",
                "augment_probability": augment_probability,
            },
        )

        return mr_train, inserter_train, misspell_train

    mr_train_2, inserter_train_2, misspell_train_2 = _augment_good_techniques(
        train_set, 0.2
    )
    mr_train_4, inserter_train_4, misspell_train_4 = _augment_good_techniques(
        train_set, 0.4
    )
    mr_train_6, inserter_train_6, misspell_train_6 = _augment_good_techniques(
        train_set, 0.6
    )
    mr_train_8, inserter_train_8, misspell_train_8 = _augment_good_techniques(
        train_set, 0.8
    )
    tensorflow_training_wrapper(mr_train_2, eval_set, test_set, f"{dataset}_mr_2")
    tensorflow_training_wrapper(mr_train_4, eval_set, test_set, f"{dataset}_mr_4")
    tensorflow_training_wrapper(mr_train_6, eval_set, test_set, f"{dataset}_mr_6")
    tensorflow_training_wrapper(mr_train_8, eval_set, test_set, f"{dataset}_mr_8")

    tensorflow_training_wrapper(mr_train_2, eval_set, test_set, f"{dataset}_mr_2")
    tensorflow_training_wrapper(mr_train_4, eval_set, test_set, f"{dataset}_mr_4")
    tensorflow_training_wrapper(mr_train_6, eval_set, test_set, f"{dataset}_mr_6")
    tensorflow_training_wrapper(mr_train_8, eval_set, test_set, f"{dataset}_mr_8")

    tensorflow_training_wrapper(
        inserter_train_2, eval_set, test_set, f"{dataset}_inserter_2"
    )
    tensorflow_training_wrapper(
        inserter_train_4, eval_set, test_set, f"{dataset}_inserter_4"
    )
    tensorflow_training_wrapper(
        inserter_train_6, eval_set, test_set, f"{dataset}_inserter_6"
    )
    tensorflow_training_wrapper(
        inserter_train_8, eval_set, test_set, f"{dataset}_inserter_8"
    )

    tensorflow_training_wrapper(
        misspell_train_2, eval_set, test_set, f"{dataset}_misspell_2"
    )
    tensorflow_training_wrapper(
        misspell_train_4, eval_set, test_set, f"{dataset}_misspell_2"
    )
    tensorflow_training_wrapper(
        misspell_train_6, eval_set, test_set, f"{dataset}_misspell_2"
    )
    tensorflow_training_wrapper(
        misspell_train_8, eval_set, test_set, f"{dataset}_misspell_2"
    )
