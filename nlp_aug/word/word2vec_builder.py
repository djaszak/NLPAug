# Download dataset
imdb_dataset = load_dataset("imdb", split="train")
data = []
t = TreebankWordTokenizer()
for sent in imdb_dataset["text"]:
    new_sent = []
    for word in t.tokenize(sent):
        # for word in nlp(sentence):
        # print(word)
        new_word = word.lower()
        if new_word[0] not in string.punctuation:
            new_sent.append(new_word)
    if len(new_sent) > 0:
        data.append(new_sent)
# model = Word2Vec(
#     sentences = data,
#     vector_size = 50,
#     window = 10,
#     epochs = 20,
# )

# model.save("word2vec.model")

model = Word2Vec.load("word2vec.model")

nlp = spacy.load("en_core_web_lg")
data_feature = imdb_dataset['text'][0]

doc = nlp(data_feature)


