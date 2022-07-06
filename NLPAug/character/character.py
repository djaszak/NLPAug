import json
import os
import random
import string

from datasets import concatenate_datasets, load_dataset
from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer
from pathlib import Path
from typing import List
from tqdm import tqdm

KEYBOARD_REPLACEMENTS = {
    "q": ["w", "s", "a"],
    "w": ["q", "e", "a", "d", "s"],
    "e": ["w", "s", "d", "f", "r"],
    "r": ["e", "d", "f", "g", "t"],
    "t": ["r", "f", "g", "h", "y"],
    "y": ["t", "g", "h", "j", "u"],
    "u": ["y", "h", "j", "k", "i"],
    "i": ["u", "j", "k", "l", "o"],
    "o": ["i", "k", "l", "p"],
    "p": ["o", "l"],
    "a": ["q", "w", "s", "x", "z"],
    "s": ["q", "a", "z", "x", "c", "d", "e", "w"],
    "d": ["w", "s", "x", "c", "v", "f", "r", "e"],
    "f": ["e", "d", "c", "v", "b", "g", "t", "r"],
    "g": ["r", "f", "v", "b", "n", "h", "y", "t"],
    "h": ["t", "g", "b", "n", "m", "j", "u", "y"],
    "j": ["y", "h", "n", "m", "k", "i", "u"],
    "k": ["u", "j", "m", "l", "o", "i"],
    "l": ["i", "k", "p", "o"],
    "z": ["a", "s", "x"],
    "x": ["a", "z", "c", "d", "s"],
    "c": ["s", "x", "v", "f", "d"],
    "v": ["d", "c", "b", "g", "f"],
    "b": ["f", "v", "n", "h", "g"],
    "n": ["g", "b", "m", "j", "h"],
    "m": ["h", "n", "k", "j"],
}


class Character:
    def __init__(self):
        working_directory = Path(os.getcwd())
        path = working_directory / "data" / "missp_data.json"
        with path.open() as f:
            self.missp = json.load(f)

    def noise_induction(self, text):
        # Won't do this now.
        pass

    @staticmethod
    def random_switcher(text: List[str]) -> List[str]:
        """
        Randomly switches two letters in word.
        Args:
            text: A string of text. If multiple words, for each word one random switch is made.

        Returns:
            A string of text with randomly switched letters.
        """
        new_words = []
        for word in text:
            if len(word) > 1:
                pos_1 = random.randint(0, len(word) - 1)
                pos_2 = random.randint(0, len(word) - 1)
                while pos_1 == pos_2:
                    pos_2 = random.randint(0, len(word) - 1)
                new_word = list(word)
                new_word[pos_1] = word[pos_2]
                new_word[pos_2] = word[pos_1]
                new_words.append("".join(new_word))
            else:
                new_words.append(word)
        return new_words

    @staticmethod
    def mid_randomizer(text: List[str]) -> List[str]:
        new_words = []
        for word in text:
            if len(word) > 3:
                rand_mid_part = random.sample(word[1:-1], len(word[1:-1]))
                rand_mid_part.insert(0, word[0])
                rand_mid_part.append(word[-1])
                new_words.append("".join(rand_mid_part))
            else:
                new_words.append(word)
        return new_words

    @staticmethod
    def complete_randomizer(text: List[str]) -> List[str]:
        new_words = []
        for word in text:
            if len(word) > 1:
                new_word = random.sample(word, len(word))
                new_words.append("".join(new_word))
            else:
                new_words.append(word)
        return new_words

    @staticmethod
    def keyboard_replacer(text: List[str]) -> List[str]:
        new_words = []
        for word in text:
            # Interesting behavior, 1-letter words are always changed.
            # if len(word) > 1:
            new_word = list(word)
            rand_index = random.randint(0, len(word) - 1)
            try:
                new_word[rand_index] = random.choice(
                    KEYBOARD_REPLACEMENTS[word[rand_index].lower()]
                )
            except KeyError:
                pass
            new_words.append("".join(new_word))
        return new_words

    @staticmethod
    def remover(text: List[str]) -> List[str]:
        new_words = []
        for word in text:
            if len(word) > 1:
                new_word = list(word)
                new_word.pop(random.randint(0, len(word) - 1))
                new_words.append("".join(new_word))
            else:
                new_words.append(word)
        return new_words

    @staticmethod
    def inserter(text: List[str]) -> List[str]:
        new_words = []
        for word in text:
            if len(word) > 1:
                new_word = list(word)
                new_word.insert(
                    random.randint(0, len(word) - 1),
                    random.choice(string.ascii_lowercase),
                )
                new_words.append("".join(new_word))
        return new_words

    def misspeller(self, text: List[str]) -> List[str]:
        new_words = []
        for word in text:
            if word in self.missp:
                new_words.append(random.choice(self.missp[word]))
            else:
                new_words.append(word)
        return new_words


# -- File augmenter CLI -- #
# augmenter = Character()
#
# parser = argparse.ArgumentParser(prog='augmenter', description='Augment data corpora with different possible .')
# parser.add_argument('in_file', type=str, help='The path to the corpus that should be augmented.')
# parser.add_argument('--modes', action='store', default='all', nargs='+',
#                     help='Which augmentation modes should be used. Provide a list of strings seperated by a comma.')
#
# in_file = parser.parse_args().in_file
# modes = parser.parse_args().modes
#
# if 'all' in modes:
#     modes = [method for method in dir(Character) if method.startswith('__') is False]
#
# new_document = []
# t = TreebankWordTokenizer()
# d = TreebankWordDetokenizer()
#
# for mode in modes:
#     with open(in_file, 'r') as f:
#         for line in tqdm(f):
#             if not line.startswith('<'):
#                 new_line = []
#                 for token in t.tokenize(line):
#                     if token.isalpha():
#                         augmented_token = getattr(augmenter, mode)([token])[0]
#                         new_line.append(augmented_token)
#                     else:
#                         new_line.append(token)
#                 new_document.append(d.detokenize(new_line) + '\n')
#
#     with open(f'augmented_output_{mode}.dat', 'w') as f:
#         f.writelines(new_document)

# -- Testing augmentation of IMDB -- #

# TODO: Add datatype
def augment_data(data, method: str):
    """

    Args:
        method: The augmentation method that should be used.
        data: Data from a specific dataset

    Returns:
        Augmented data.
    """
    t = TreebankWordTokenizer()
    d = TreebankWordDetokenizer()
    augmenter = Character()

    new_line = []
    # print(data)
    try:
        for token in t.tokenize(data):
            if token.isalpha():
                augmented_token = getattr(augmenter, method)([token])[0]
                new_line.append(augmented_token)
            else:
                new_line.append(token)
    except TypeError:
        print(type(data))
    data = d.detokenize(new_line)

    return data


def misspell_data(data):
    data["text"] = augment_data(data["text"], "misspeller")
    return data


def random_switcher_data(data):
    data["text"] = augment_data(data["text"], "random_switcher")
    return data


def mid_randomizer_data(data):
    data["text"] = augment_data(data["text"], "mid_randomizer")
    return data


def complete_randomizer_data(data):
    data["text"] = augment_data(data["text"], "complete_randomizer")
    return data


def keyboard_replacer_data(data):
    data["text"] = augment_data(data["text"], "keyboard_replacer")
    return data


def remover_data(data):
    data["text"] = augment_data(data["text"], "remover")
    return data


def inserter_data(data):
    data["text"] = augment_data(data["text"], "inserter")
    return data


TRAIN = "train"
TEST = "test"
UNNAMED = "Unnamed: 0"
# An example I wanna use

imdb_train = load_dataset("imdb", split="train")
imdb_test = load_dataset("imdb", split="test")
#
# imdb_train.map(misspell_data, num_proc=6).to_csv(
#     '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/misspelled_imdb_train.csv')
# imdb_test.map(misspell_data, num_proc=6).to_csv(
#     '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/misspelled_imdb_test.csv')
#
# imdb_train.map(random_switcher_data, num_proc=6).to_csv(
#     '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/random_switcher_imdb_train.csv')
# imdb_test.map(random_switcher_data, num_proc=6).to_csv(
#     '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/random_switcher_imdb_test.csv')
#
# imdb_train.map(mid_randomizer_data, num_proc=6).to_csv(
#     '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/mid_randomizer_imdb_train.csv')
# imdb_test.map(mid_randomizer_data, num_proc=6).to_csv(
#     '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/mid_randomizer_imdb_test.csv')
#
# imdb_train.map(complete_randomizer_data, num_proc=6).to_csv(
#     '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/complete_randomizer_imdb_train.csv')
# imdb_test.map(complete_randomizer_data, num_proc=6).to_csv(
#     '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/complete_randomizer_imdb_test.csv')
#
# imdb_train.map(keyboard_replacer_data, num_proc=6).to_csv(
#     '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/keyboard_replacer_imdb_train.csv')
# imdb_test.map(keyboard_replacer_data, num_proc=6).to_csv(
#     '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/keyboard_replacer_imdb_test.csv')

imdb_train.map(remover_data, num_proc=6).to_csv(
    "/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/remover_imdb_train.csv"
)
imdb_test.map(remover_data, num_proc=6).to_csv(
    "/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/remover_imdb_test.csv"
)

imdb_train.map(inserter_data, num_proc=6).to_csv(
    "/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/inserter_imdb_train.csv"
)
imdb_test.map(inserter_data, num_proc=6).to_csv(
    "/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/inserter_imdb_test.csv"
)
# print(misspelled_dataset[TEST])

# dataset = load_dataset('csv', data_files={'train': '/home/dennis/Uni/GrosserBeleg/augmented_imdb_datasets/misspelled_imdb_train.csv'})
# print(dataset.remove_columns(UNNAMED))
