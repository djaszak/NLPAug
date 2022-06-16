import json

files = ['cs.key', 'de.key', 'en.key', 'fr.key', 'cs.natural', 'de.natural', 'en.natural', 'fr.natural']
files = ['de.natural']

noise_dict = {}

for file in files:
    with open(file, 'r') as f:
        for line in f:
            line_list = line.split(' ')
            noise_dict[line_list[0]] = line_list[1:]

    with open(f'{file}.json', 'w') as f:
        json.dump(noise_dict, f)


