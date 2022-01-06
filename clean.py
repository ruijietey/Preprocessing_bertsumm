import json

RESULT_PATH = "results/"
text_file = RESULT_PATH + "CNNDM_distilbert.jsonl"
index_file = RESULT_PATH + "index.jsonl"

original = []
indices = []

with open(text_file, "r") as f:
    lines = f.readlines()
    count = 0
    for i, line in enumerate(lines):
        if line != "\n":
            # print(line)
            count += 1
            try:
                original.append(json.loads(line))
            except:
                print(f'Line no: {i+1}, {count}th document')
                # print(f'Content: {line}')

    print(count)

# print(original)

with open(index_file, "r") as f:
    lines = f.readlines()
    count = 0
    for i, line in enumerate(lines):
        if line != "\n":
            # print(line)
            count += 1
            try:
                indices.append(json.loads(line))
            except:
                print(line)
                print(f'Line no: {i+1}, {count}th document')
                # print(f'Content: {line}')

    print(count)
