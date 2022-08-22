import random

from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer


def augment_data(data, augmenter, method: str, augment_probability: float = 1):
    """

    Args:
        method: The augmentation method that should be used.
        data: Data from a specific dataset

    Returns:
        Augmented data.
    """
    t = TreebankWordTokenizer()
    d = TreebankWordDetokenizer()

    new_line = []
    try:
        for token in t.tokenize(data):
            if token.isalpha() and augment_probability >= random.random():
                augmented_token = getattr(augmenter, method)([token])
                try:
                    new_line.append(augmented_token[0])
                except IndexError:
                    pass
            else:
                new_line.append(token)
    except TypeError:
        print(type(data))
    data = d.detokenize(new_line)

    return data
