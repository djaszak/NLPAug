parsed_dict = {}

with open('missp.dat', 'r') as f:
    for line in f:
        if line.startswith('$'):
            parsed_dict[line.strip('$ ')] = []
