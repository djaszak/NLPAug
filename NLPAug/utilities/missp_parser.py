import json

parsed_dict = {}

with open("missp.dat", "r") as f:
    for line in f:
        if line.startswith("$"):
            current_correct = line.strip("$ ").replace("\n", "")
            parsed_dict[current_correct] = []
        else:
            parsed_dict[current_correct].append(line.replace("\n", ""))

with open("../character/data/missp_data.json", "w") as f:
    json.dump(parsed_dict, f)
