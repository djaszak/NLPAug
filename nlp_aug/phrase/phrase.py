import spacy

from nltk.tokenize import TreebankWordDetokenizer

from spacy.matcher import Matcher


class PhraseReplIns:
    """A class working as an interface for operations that are aiming at either replacing or inserting words by defined rules."""

    thesaurus = None

    def __init__(self, nlp=None):
        if not nlp:
            self.nlp = spacy.load("en_core_web_lg")
        self.matcher = Matcher(self.nlp.vocab)

    # def replacement_rule(self, token: spacy.tokens.token.Token) -> bool:
    #     """Implement the rule used to replace as well as how to handle the given thesaurus."""
    #     NotImplementedError
    #     pass

    # def candidate_selection(self, token: spacy.tokens.token.Token) -> str:
    #     """Implement the statistical approach which specific synonym from a list should be used."""
    #     pass

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

        new_doc = []

        for token in doc:
            if self.replacement_rule(token):
                replacement = self.candidate_selection(token)
                replacement = replacement.replace("_", " ")
                new_doc.append(replacement)
            else:
                new_doc.append(token.text)

        d = TreebankWordDetokenizer()
        data[augmented_feature] = d.detokenize(new_doc)
        return data
