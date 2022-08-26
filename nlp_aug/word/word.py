from datasets import (
    load_dataset,
    concatenate_datasets,
)
import random
import spacy

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordDetokenizer
from spacy.matcher import Matcher


class Word:
    def unigram_noising():
        pass

    def blank_noising(text: list) -> list:
        return ["_" for word in text]

    def syntactic_noising():
        pass

    def semantic_noising():
        pass

    def random_swap():
        pass

    def random_delete():
        pass

    def synonym_replacer():
        pass


class SynonymReplacer:
    thesaurus = None

    def __init__(self, thesaurus=None):
        self.thesaurus = thesaurus
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)

    def replacement_rule(self, token: spacy.tokens.token.Token) -> bool:
        """Implement the rule used to replace as well as how to handle the given thesaurus."""
        pass

    def synonym_selection(self, token: spacy.tokens.token.Token) -> str:
        """Implement the statistical approach which specific synonym from a list should be used."""
        pass

    # https://stackoverflow.com/questions/62785916/spacy-replace-token
    def replace_word(self, orig_text, replacement):
        tok = self.nlp(orig_text)
        text = ""
        buffer_start = 0
        for _, match_start, _ in self.matcher(tok):
            if (
                match_start > buffer_start
            ):  # If we've skipped over some tokens, let's add those in (with trailing whitespace if available)
                text += (
                    tok[buffer_start:match_start].text
                    + tok[match_start - 1].whitespace_
                )
            text += (
                replacement + tok[match_start].whitespace_
            )  # Replace token, with trailing whitespace if available
            buffer_start = match_start + 1
        if buffer_start < len(tok):
            text += tok[buffer_start:].text
        return text

    def engine(self, data: str) -> str:
        """Using `replacement_rule()` and `synonym_selection()` a string is augmented

        Args:
            data (str): A list consisting out of strings that are representing a list of natural language
                sentences. Every sentence is augmented in a way defined by the `replacement_rule()` and
                the `synonym_selection()`.

        Returns:
            str: The augmented string.
        """

        doc = self.nlp(data)

        new_doc = []

        for token in doc:
            if self.replacement_rule(token):
                replacement = self.synonym_selection(token)
                replacement = replacement.replace("_", " ")
                new_doc.append(self.synonym_selection(token))
            else:
                new_doc.append(token.text)

        d = TreebankWordDetokenizer()

        return d.detokenize(new_doc)


class BaseReplacer(SynonymReplacer):
    def replacement_rule(
        self, token: spacy.tokens.token.Token, replacement_prob: float = 0.5
    ) -> bool:
        allowed_pos = ["NOUN", "VERB", "ADJ", "ADV"]

        if token.pos_ in allowed_pos and replacement_prob >= random.random():
            return True

        return False

    def synonym_selection(self, token: spacy.tokens.token.Token) -> str:
        synonyms = []

        if token.is_alpha and not token.is_stop:
            synsets = wn.synsets(token.lemma_, pos=getattr(wn, token.pos_))
            for syn in synsets:
                for lm in syn.lemmas():
                    synonyms.append(lm.name())

            synonyms.remove(token.text)

        try:
            return random.choice(list(set(synonyms)))
        except IndexError:
            return ""


replacer = BaseReplacer()

print(
    replacer.engine(
        "I didn't watch the news yesterday, I read the paper. It's pretty nice outside."
    )
)

imdb_dataset = load_dataset("imdb", split='train')
cr_train = (
    imdb_dataset
    # .select(range(100))
    .map(
        replacer.engine,
        num_proc=4,
    )
)

print(cr_train['text'])
