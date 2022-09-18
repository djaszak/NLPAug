class Word2VecBuilder:
    def __init__(data):
        self.data = data

    def build(name: str):
        data = []
        t = TreebankWordTokenizer()
        for sent in data:
            new_sent = []
            for word in t.tokenize(sent):
                new_word = word.lower()
                if new_word[0] not in string.punctuation:
                    new_sent.append(new_word)
            if len(new_sent) > 0:
                data.append(new_sent)
                
        model = Word2Vec(
            sentences = data,
            vector_size = 50,
            window = 10,
            epochs = 20,
        )

        model.save(f"{name}.model")
