from nlp_aug.trainer.training_utils import tensorflow_training_wrapper
from nlp_aug.trainer.data_builder import (
    imdb_eval,
    imdb_test,
    imdb_train,
    cr_train,
    kr_train,
    mr_train,
    rs_train,
    inserter_train,
    remover_train,
    misspell_train,
    cr_imdb_train,
    kr_imdb_train,
    mr_imdb_train,
    rs_imdb_train,
    inserter_imdb_train,
    remover_imdb_train,
    misspell_imdb_train,
)

history, model, evaluation = tensorflow_training_wrapper(
    imdb_train, imdb_eval, "imdb", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    cr_train, imdb_eval, "cr", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    kr_train, imdb_eval, "kr", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    mr_train, imdb_eval, "mr", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    rs_train, imdb_eval, "rs", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    inserter_train, imdb_eval, "inserter", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    remover_train, imdb_eval, "remover", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    misspell_train, imdb_eval, "misspell", imdb_test
)

history, model, evaluation = tensorflow_training_wrapper(
    cr_imdb_train, imdb_eval, "cr_imdb", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    kr_imdb_train, imdb_eval, "kr_imdb", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    mr_imdb_train, imdb_eval, "mr_imdb", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    rs_imdb_train, imdb_eval, "rs_imdb", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    inserter_imdb_train, imdb_eval, "inserter_imdb", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    remover_imdb_train, imdb_eval, "remover_imdb", imdb_test
)
history, model, evaluation = tensorflow_training_wrapper(
    misspell_imdb_train, imdb_eval, "misspell_imdb", imdb_test
)
