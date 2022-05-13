import random
import string
from typing import List

KEYBOARD_REPLACEMENTS = {
    'q': ['w', 's', 'a'],
    'w': ['q', 'e', 'a', 'd', 's'],
    'e': ['w', 's', 'd', 'f', 'r'],
    'r': ['e', 'd', 'f', 'g', 't'],
    't': ['r', 'f', 'g', 'h', 'y'],
    'y': ['t', 'g', 'h', 'j', 'u'],
    'u': ['y', 'h', 'j', 'k', 'i'],
    'i': ['u', 'j', 'k', 'l', 'o'],
    'o': ['i', 'k', 'l', 'p'],
    'p': ['o', 'l'],
    'a': ['q', 'w', 's', 'x', 'z'],
    's': ['q', 'a', 'z', 'x', 'c', 'd', 'e', 'w'],
    'd': ['w', 's', 'x', 'c', 'v', 'f', 'r', 'e'],
    'f': ['e', 'd', 'c', 'v', 'b', 'g', 't', 'r'],
    'g': ['r', 'f', 'v', 'b', 'n', 'h', 'y', 't'],
    'h': ['t', 'g', 'b', 'n', 'm', 'j', 'u', 'y'],
    'j': ['y', 'h', 'n', 'm', 'k', 'i', 'u'],
    'k': ['u', 'j', 'm', 'l', 'o', 'i'],
    'l': ['i', 'k', 'p', 'o'],
    'z': ['a', 's', 'x'],
    'x': ['a', 'z', 'c', 'd', 's'],
    'c': ['s', 'x', 'v', 'f', 'd'],
    'v': ['d', 'c', 'b', 'g', 'f'],
    'b': ['f', 'v', 'n', 'h', 'g'],
    'n': ['g', 'b', 'm', 'j', 'h'],
    'm': ['h', 'n', 'k', 'j'],
}


class Character:
    def __init__(self):
        pass

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
                new_words.append(''.join(new_word))
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
                new_words.append(''.join(rand_mid_part))
            else:
                new_words.append(word)
        return new_words

    @staticmethod
    def complete_randomizer(text: List[str]) -> List[str]:
        new_words = []
        for word in text:
            if len(word) > 1:
                new_word = random.sample(word, len(word))
                new_words.append(''.join(new_word))
            else:
                new_words.append(word)
        return new_words

    @staticmethod
    def keyboard_replacer(text: List[str]) -> List[str]:
        new_words = []
        for word in text:
            # Interesting behavior, 1 letter words are always changed.
            # if len(word) > 1:
            new_word = list(word)
            rand_index = random.randint(0, len(word) - 1)
            new_word[rand_index] = random.choice(KEYBOARD_REPLACEMENTS[word[rand_index]])
            new_words.append(''.join(new_word))
        return new_words

    @staticmethod
    def remover(text: List[str]) -> List[str]:
        new_words = []
        for word in text:
            if len(word) > 1:
                new_word = list(word)
                new_word.pop(random.randint(0, len(word) - 1))
                new_words.append(''.join(new_word))
        return new_words

    @staticmethod
    def inserter(text: List[str]) -> List[str]:
        new_words = []
        for word in text:
            if len(word) > 1:
                new_word = list(word)
                new_word.insert(random.randint(0, len(word) - 1), random.choice(string.ascii_lowercase))
                new_words.append(''.join(new_word))
        return new_words


for word in Character.inserter('Hello world good morning fine worlds and you and me all is super goood'):
    print(word)
