from hashlib import new
from lib2to3.pytree import Base
from re import sub
import spacy

from colorama import Fore, Style

from datasets import load_dataset

from nltk.tokenize import TreebankWordDetokenizer

from spacy import displacy
from spacy.matcher import Matcher


class PhraseAugmenter:
    """A class working as an interface for operations that are aiming at either replacing or inserting words by defined rules."""

    thesaurus = None

    def __init__(self, nlp=None):
        if not nlp:
            self.nlp = spacy.load("en_core_web_lg")
        self.matcher = Matcher(self.nlp.vocab)

    def transformation(self, sent: str) -> str:
        """Implement the transformation that should be done to the sentence."""
        return sent


    def engine(self, data: str,) -> str:
        """Using `replacement_rule()` and `synonym_selection()` a string is augmented

        Args:
            data (str): A list consisting out of strings that are representing a list of natural language
                sentences. Every sentence is augmented in a way defined by the `replacement_rule()` and
                the `synonym_selection()`.

        Returns:
            str: The augmented string.
        """
        data_feature = data
        doc = self.nlp(data_feature)
        sents = doc.sents

        new_doc = []

        for sent in sents:
            new_doc.append(self.transformation(sent))

        data = '. '.join(new_doc)

        return data

class BaseCropper(PhraseAugmenter):
    def transformation(self, sent: str, dependency_focus: str = 'nsubj') -> str:
        sent = super().transformation(sent)

        subj_list = [token for token in sent if token.dep_ == dependency_focus]

        head_dict = {}

        for subj in subj_list:
            head = subj.head
            head_dict[subj] = [head]
            while head != head.head:
                head_dict[subj].append(head)
                head = head.head
            head_dict[subj] = list(set(head_dict[subj]))
        
        new_sent = []
        for key, val in head_dict.items(): 
            new_sent.append([key, val]) 

        real_new_sent = []
        for part in new_sent:
            real_new_sent.append(part[0].text)
            for smaller_part in part[1]:
                real_new_sent.append(smaller_part.text)
            real_new_sent.append(',')

        real_new_sent[0] = real_new_sent[0].capitalize()
        real_new_sent.pop(-1)
        d = TreebankWordDetokenizer()
        sent = d.detokenize(real_new_sent)

        return sent

class BaseRotation(PhraseAugmenter):
    def transformation(self, sent: str, dependency_focus: str = 'nsubj') -> str:
        sent = super().transformation(sent)

        root = [token for token in sent if token.dep_ == dependency_focus][0].text
        sent = list(sent)
        first_part = []
        for x, word in enumerate(sent):
            word = word.text
            if word == root:
                break
            first_part.append(word)
            sent.pop(x)

        second_part = [token.text for token in sent]
        new_sent = []
        for word in second_part:
            new_sent.append(word)
        new_sent.append(root)
        for word in first_part:
            new_sent.append(word)

        d = TreebankWordDetokenizer()
        sent = d.detokenize(new_sent)

        return sent

imdb_dataset = load_dataset("imdb", split="train").select(range(1))
cropper = BaseRotation()
for data in imdb_dataset:
    print(data['text'])
    print(cropper.engine(data['text']))
