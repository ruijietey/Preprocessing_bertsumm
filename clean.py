import json

RESULT_PATH = "results/"
text_file = RESULT_PATH + "CNNDM_distilbert.jsonl"
index_file = RESULT_PATH + "index.jsonl"

original = []
indices = []

passed_set = set()
trace_doc_map = {}

with open(text_file, "r") as f:
    lines = f.readlines()
    count = 0
    for i, line in enumerate(lines):
        if line != "\n":
            count += 1
            try:
                obj = json.loads(line)
                original.append(obj)
                passed_set.add(obj["id"])
            except:
                print(f'Text Line no: {i+1}, {count}th document')
                print(f'Content: {line}')

    print(len(original))

with open(index_file, "r") as f:
    lines = f.readlines()
    count = 0
    for i, line in enumerate(lines):
        if line != "\n":
            count += 1
            try:
                obj = json.loads(line)
                if obj["id"] in passed_set:
                    indices.append(obj)
            except:
                print(line)
                print(f'Index Line no: {i+1}, {count}th document')
                # print(f'Content: {line}')

for i, sent_id in enumerate(indices):
    if sent_id["id"] in passed_set:
        trace_doc_map[sent_id["id"]] = i

cleaned_original = []
cleaned_indices = []


for doc in original:
    if doc["id"] in trace_doc_map:
        try:
            cleaned_original.append(doc)
            cleaned_indices.append(indices[trace_doc_map[doc["id"]]])
        except:
            print(doc)
            print(trace_doc_map[doc["id"]])


for i in range(len(cleaned_original)):
    if cleaned_original[i]["id"] != cleaned_indices[i]["id"]:
        print(f'{i} index mismatched')

# for cleaned_doc in

