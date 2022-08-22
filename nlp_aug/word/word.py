from nltk.corpus import wordnet as wn

class Word:
    def unigram_noising():
        pass

    def blank_noising(text: list) -> list:
        return ['_' for word in text]

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

    def __init__(thesaurus):
        thesaurus = thesaurus

    def replacement_rule():
        """Implement the rule used to replace as well as how to handle the given thesaurus.
        """
        pass

    def synonym_selection():
        """Implement the statistical approach which specific synonym from a list should be used.
        """
        pass

    def engine(text: list) -> list:
        """Using `replacement_rule()` and `synonym_selection()` a list containing strings, consisting out
        of sentences is augmented

        Args:
            text (list): A list consisting out of strings that are representing a list of natural language
                sentences. Every sentence is augmented in a way defined by the `replacement_rule()` and 
                the `synonym_selection()`.

        Returns:
            list: The augmented list of sentences.
        """
        pass

