from bs4 import BeautifulSoup
from colorama import Fore, Style
from datasets import load_dataset
from gensim.models import Word2Vec

from nlp_aug.character.character import Character
from nlp_aug.word.word import BaseSynonymReplIns, BaseEmbeddingReplIns
from nlp_aug.word.word2vec_builder import Word2VecBuilder
from nlp_aug.utilities.augment_utils import augment_data

## At first we get a real life dataset, which I load from huggingface
## For this demo, one sample is more than enough
imdb_dataset = load_dataset("imdb", split="train").select(range(1))

## Prepare the imdb_data to get one data sample which will be augmented
imdb_text = imdb_dataset["text"][0]
soup = BeautifulSoup(imdb_text, "html.parser")
imdb_text = " ".join(soup.stripped_strings)
print(f"{Fore.BLUE}Basic data:{Style.RESET_ALL} {imdb_text}")
character_augmenter = Character()

## We start by using all implemented modes of the character augmentation
modes = [
    "random_switcher",
    "mid_randomizer",
    "complete_randomizer",
    "keyboard_replacer",
    "remover",
    "inserter",
    "misspeller",
]

for mode in modes:
    augmented_data_string = augment_data(imdb_text, character_augmenter, mode, 0.5)
    print(f"{Fore.BLUE}Using mode {mode}:{Style.RESET_ALL} {augmented_data_string}")

## Now we use the word level augmenters to do some synonym and embedding replacements
# synonym_replaced_text = BaseSynonymReplIns().replace_engine(imdb_text)
# synonym_inserted_text = BaseSynonymReplIns().insert_engine(imdb_text)

# ## Build a Word2Vec Model to use WordEmbeddings
# Word2VecBuilder(imdb_dataset["text"]).build("demo_word2vec")

# model = Word2Vec.load("word2vec.model")

# embedding_replaced_text = BaseEmbeddingReplIns(word2vec=model).replace_engine(imdb_text)
# embedding_inserted_text = BaseEmbeddingReplIns(word2vec=model).insert_engine(imdb_text)

# print(
#     f"{Fore.BLUE}Using augmentation method synonym replacement:{Style.RESET_ALL} {synonym_replaced_text}"
# )
# print(
#     f"{Fore.BLUE}Using augmentation method synonym insertion:{Style.RESET_ALL} {synonym_inserted_text}"
# )
# print(
#     f"{Fore.BLUE}Using augmentation method embedding Ich freue mich schon jetzt auf die Dusche @Manuel ich denke du kannst dich für morgen austragen 💁‍♂️😘 (sorry)replacement:{Style.RESET_ALL} {embedding_replaced_text}"
# )
# print(
#     f"{Fore.BLUE}Using augmentation method :{Style.RESET_ALL} {embedding_inserted_text}"
# )
