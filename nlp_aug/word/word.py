import random
import spacy
from bs4 import BeautifulSoup
from colorama import Fore, Style
from datasets import load_dataset
from gensim.models import Word2Vec
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordDetokenizer
from spacy.matcher import Matcher
from nlp_aug import constants
from nlp_aug.word.word2vec_builder import Word2VecBuilder


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

    def replace_engine(self, data: str, replacement_prob: float = 0.5) -> str:
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
            if self.replacement_rule(token, replacement_prob=replacement_prob):
                replacement = self.candidate_selection(token)
                replacement = replacement.replace("_", " ")
                new_doc.append(replacement)
            else:
                new_doc.append(token.text)

        d = TreebankWordDetokenizer()
        data = d.detokenize(new_doc)
        return data

    def insert_engine(self, data: str, replacement_prob: float = 0.5) -> str:
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
            if self.replacement_rule(token, replacement_prob=replacement_prob):
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


class BaseReplIns(WordReplIns):
    def replacement_rule(
        self, token: spacy.tokens.token.Token, replacement_prob: float = 0.5
    ) -> bool:
        allowed_pos = ["NOUN", "VERB", "ADJ", "ADV"]

        if token.pos_ in allowed_pos and replacement_prob >= random.random():
            return True

        return False


class BaseSynonymReplIns(BaseReplIns):
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
                pass
                # print("Lemma: ", token.lemma_, " not found in ", synonyms)
        try:
            return random.choice(list(set(synonyms)))
        except IndexError:
            return token.text


class BaseEmbeddingReplIns(BaseReplIns):
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


#### Here starts the new augmenter with using the stuff from the demo
# imdb_dataset = load_dataset("imdb", split="train").select(range(1))

# imdb_text = imdb_dataset["text"][0]
# soup = BeautifulSoup(imdb_text, "html.parser")
# imdb_text = " ".join(soup.stripped_strings)
# print(f"{Fore.BLUE}Basic data:{Style.RESET_ALL} {imdb_text}")

# # Now we use the word level augmenters to do some synonym and embedding replacements
# synonym_replaced_text = BaseSynonymReplIns().replace_engine(imdb_text)
# synonym_inserted_text = BaseSynonymReplIns().insert_engine(imdb_text)

# ## Build a Word2Vec Model to use WordEmbeddings
# Word2VecBuilder(imdb_dataset["text"]).build("demo_word2vec")

# model = Word2Vec.load("word2vec.model")

# embedding_replaced_text = BaseEmbeddingReplIns(word2vec=model).replace_engine(imdb_text)
# embedding_inserted_text = BaseEmbeddingReplIns(word2vec=model).insert_engine(imdb_text)

# Word2VecBuilder(imdb_dataset["text"]).build("demo_word2vec")

# model = Word2Vec.load("word2vec.model")

#### Here comes the real new code, all above is just demo

def augment_data(data, mode, augment_probability, word2vec_model = None):
    """Augment data on word level. 4 modes will be supported

    Args:
        data: Just some strings
        mode: 4 modes are supported:
            * synonym_replacement
            * synonym_inserter
            * embedding_replacement
            * embedding_inserter
        augment_probability: _description_
    """
    # data = data
    augmented_text = ''
    if mode == constants.SYNONYM_INSERTER:
        augmented_text = BaseSynonymReplIns().insert_engine(data, replacement_prob=augment_probability)
    if mode == constants.SYNONYM_REPLACEMENT:
        augmented_text = BaseSynonymReplIns().replace_engine(data, replacement_prob=augment_probability)
    if mode == constants.EMBEDDING_INSERTER or mode == constants.EMBEDDING_REPLACEMENT:
        if mode == constants.EMBEDDING_INSERTER:
            augmented_text = BaseEmbeddingReplIns(word2vec=model).insert_engine(data, replacement_prob=augment_probability)
        if mode == constants.EMBEDDING_REPLACEMENT:
            augmented_text = BaseEmbeddingReplIns(word2vec=model).replace_engine(data, replacement_prob=augment_probability)

    return augmented_text

def word_augment_huggingface_data(
    data, augmented_feature: str, mode: str, augment_probability: float = 1, word2vec_model = None
):
    data[augmented_feature] = augment_data(
        data[augmented_feature],
        mode,
        augment_probability=augment_probability,
        word2vec_model
    )
    return data
