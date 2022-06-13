import argparse

from NLPAug.character.character import Character

parser = argparse.ArgumentParser(description='Augment data corpora with different possible .')
parser.add_argument('in_file', type=str, help='The path to the corpus that should be augmented.')
parser.add_argument('--modes', action='store', default='all',
                    help='Which augmentation modes should be used. Provide a list of string seperated by a comma.')

in_file = parser.parse_args().in_file
modes = parser.parse_args().modes

augmenter = Character()

mode_list = [method for method in dir(Character) if method.startswith('__') is False]
print(mode_list)


