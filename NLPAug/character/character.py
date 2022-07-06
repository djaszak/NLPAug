import json
import os
import random
import string

from datasets import load_dataset
from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer
from pathlib import Path

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
    def random_switcher(text: str) -> str:
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
    def mid_randomizer(text: str) -> str:
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
    def complete_randomizer(text: str) -> str:
        new_words = []
        for word in text:
            if len(word) > 1:
                new_word = random.sample(word, len(word))
                new_words.append("".join(new_word))
            else:
                new_words.append(word)
        return new_words

    @staticmethod
    def keyboard_replacer(text: str) -> str:
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
    def remover(text: str) -> str:
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
    def inserter(text: str) -> str:
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

    def misspeller(self, text: str) -> str:
        new_words = []
        for word in text:
            if word in self.missp:
                new_words.append(random.choice(self.missp[word]))
            else:
                new_words.append(word)
        return new_words


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
                augmented_token = getattr(augmenter, method)(token)
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
