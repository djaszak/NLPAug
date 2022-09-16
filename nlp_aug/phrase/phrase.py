from lib2to3.pytree import Base
from re import sub
import spacy

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


    def engine(self, data: str, augmented_feature: str) -> str:
        """Using `replacement_rule()` and `synonym_selection()` a string is augmented

        Args:
            data (str): A list consisting out of strings that are representing a list of natural language
                sentences. Every sentence is augmented in a way defined by the `replacement_rule()` and
                the `synonym_selection()`.

        Returns:
            str: The augmented string.
        """
        data_feature = data[augmented_feature]
        doc = self.nlp(data_feature)
        sents = doc.sents

        new_doc = []

        for sent in sents:
            new_doc.append(self.transformation(sent))

        data[augmented_feature] = new_doc

        return data

class BaseCropper(PhraseAugmenter):
    def transformation(self, sent: str, dependency_focus: str = 'nsubj') -> str:
        sent = super().transformation(sent)

        subj_list = [token for token in sent if token.dep_ == dependency_focus]
        head_list = [token.head for token in subj_list]
        
        print(subj_list)
        print(head_list)

        return sent

imdb_dataset = load_dataset("imdb", split="train").select(range(10))
cropper = BaseCropper()
for data in imdb_dataset:
    print(cropper.engine(data, augmented_feature='text'))
