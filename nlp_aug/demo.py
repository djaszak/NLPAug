from bs4 import BeautifulSoup
from colorama import Fore, Style
from datasets import load_dataset

from nlp_aug.character.character import Character
from nlp_aug.utilities.augment_utils import augment_data

## At first we get a real life dataset, which I load from huggingface
## For this demo, one sample is more than enough
imdb_dataset = load_dataset("imdb", split="train").select(range(1))

## Starting with character augmentation
imdb_text = imdb_dataset["text"][0]
soup = BeautifulSoup(imdb_text, "html.parser")
imdb_text = " ".join(soup.stripped_strings)
print(imdb_text)
character_augmenter = Character()

# This are all available modes
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
