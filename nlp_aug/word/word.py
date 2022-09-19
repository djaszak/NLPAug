import random
import spacy

from datasets import load_dataset

from gensim.models import Word2Vec

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


class WordReplIns:
    """A class working as an interface for operations that are aiming at either replacing or inserting words by defined rules."""

    thesaurus = None

    def __init__(self, nlp=None, thesaurus=None):
        self.thesaurus = thesaurus
        if not nlp:
            self.nlp = spacy.load("en_core_web_lg")
        self.matcher = Matcher(self.nlp.vocab)

    def replacement_rule(self, token: spacy.tokens.token.Token) -> bool:
        """Implement the rule used to replace as well as how to handle the given thesaurus."""
        NotImplementedError
        pass

    def candidate_selection(self, token: spacy.tokens.token.Token) -> str:
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

    def replace_engine(self, data: str) -> str:
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

        new_doc = []

        for token in doc:
            if self.replacement_rule(token):
                replacement = self.candidate_selection(token)
                replacement = replacement.replace("_", " ")
                new_doc.append(replacement)
            else:
                new_doc.append(token.text)

        d = TreebankWordDetokenizer()
        data = d.detokenize(new_doc)
        return data

    def insert_engine(self, data: str) -> str:
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

        new_doc = []
        insertion_list = []

        for token in doc:
            if self.replacement_rule(token):
                replacement = self.candidate_selection(token)
                replacement = replacement.replace("_", " ")
                insertion_list.append(replacement)
                new_doc.append(token.text)
            else:
                new_doc.append(token.text)

        for insert in insertion_list:
            new_doc.insert(random.randint(0, len(new_doc)), insert)

        d = TreebankWordDetokenizer()
        data = d.detokenize(new_doc)
        return data


class BaseReplacer(WordReplIns):
    def replacement_rule(
        self, token: spacy.tokens.token.Token, replacement_prob: float = 0.5
    ) -> bool:
        allowed_pos = ["NOUN", "VERB", "ADJ", "ADV"]

        if token.pos_ in allowed_pos and replacement_prob >= random.random():
            return True

        return False


class BaseSynonymReplacer(BaseReplacer):
    def candidate_selection(self, token: spacy.tokens.token.Token) -> str:
        synonyms = []

        if token.is_alpha and not token.is_stop:
            # Only getting the same POS
            synsets = wn.synsets(token.lemma_, pos=getattr(wn, token.pos_))
            for syn in synsets:
                for lm in syn.lemmas():
                    synonyms.append(lm.name())
            try:
                synonyms.remove(token.lemma_)
            except ValueError:
                print("Lemma: ", token.lemma_, " not found in ", synonyms)
        try:
            return random.choice(list(set(synonyms)))
        except IndexError:
            return token.text


class BaseEmbeddingReplacer(BaseReplacer):
    def __init__(self, thesaurus=None, word2vec: Word2Vec = None):
        super().__init__(thesaurus)
        self.word2vec = word2vec

    def candidate_selection(self, token: spacy.tokens.token.Token) -> str:
        if token.is_alpha and not token.is_stop:
            # For this simple demo implementation, we are just using the most_similar.
            try:
                return self.word2vec.wv.most_similar(token.text)[0][0]
            except Exception as e:
                return token.text
        
        return token.text

model = Word2Vec.load("word2vec.model")
replacer = BaseEmbeddingReplacer(word2vec=model)


imdb_dataset = load_dataset("imdb", split="train").select(range(1))
cr_train = imdb_dataset.map(
    replacer.insert_engine,
    num_proc=4,
    fn_kwargs={'augmented_feature': 'text'}
)

print(cr_train["text"])
