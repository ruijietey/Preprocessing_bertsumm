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

with open(index_file, "r") as f:
    lines = f.readlines()
    count = 0
    for i, line in enumerate(lines):
        if line != "\n":
            # print(line)
            count += 1
            if count!= 291 and count!=2537 and count!= 2544:
                try:
                    indices.append(json.loads(line))
                except:
                    print(line)
                    print(f'Line no: {i+1}, {count}th document')
                    # print(f'Content: {line}')

    print(count)

with open(RESULT_PATH+"cleaned.jsonl", "w") as f:
    for sent in original:
        print(json.dumps(sent), file=f)

with open(RESULT_PATH+"indice.jsonl", "w") as f:
    for sent_id in indices:
        print(json.dumps(sent_id), file=f)