import json

RESULT_PATH = "results/"
USE_TRAIN_TEST_VAL = True
TRAIN_RATIO = 0.8  # Train-test-val ratio, validation ratio = test ratio = (1-TRAIN_RATIO)/2

validation_ratio = (1 - TRAIN_RATIO)/2
test_ratio = validation_ratio
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
            # Check if there is mismatch
            print(doc)
            print(trace_doc_map[doc["id"]])

# Check if there is mismatch between
assert len(cleaned_original) == len(cleaned_indices)
for i in range(len(cleaned_original)):
    if cleaned_original[i]["id"] != cleaned_indices[i]["id"]:
        print(f'{i} index mismatched')
    assert cleaned_original[i]["id"] == cleaned_indices[i]["id"]

val_data_start = int(len(cleaned_original) * TRAIN_RATIO)
test_data_start = val_data_start + int(len(cleaned_original) * validation_ratio)

print(f'Length of cleaned data: {len(cleaned_original)}')
print(f'val start: {val_data_start}')
print(f'test start: {test_data_start}')

# Saving train, test, val text and summary data
with open(RESULT_PATH+"train_CNNDM_distilbert.jsonl", "w") as f:
    for i in range(val_data_start):
        print(json.dumps(cleaned_original[i]), file=f)

with open(RESULT_PATH+"val_CNNDM_distilbert.jsonl", "w") as f:
    for i in range(val_data_start, test_data_start):
        print(json.dumps(cleaned_original[i]), file=f)

with open(RESULT_PATH+"test_CNNDM_distilbert.jsonl", "w") as f:
    for i in range(test_data_start, len(cleaned_original)):
        print(json.dumps(cleaned_original[i]), file=f)

# Cleaning index.jsonl for train,val,test
with open(RESULT_PATH+"train_indices.jsonl", "w") as f:
    for i in range(val_data_start):
        print(json.dumps(cleaned_indices[i]), file=f)

with open(RESULT_PATH+"val_indices.jsonl", "w") as f:
    for i in range(val_data_start, test_data_start):
        print(json.dumps(cleaned_indices[i]), file=f)

with open(RESULT_PATH+"test_indices.jsonl", "w") as f:
    for i in range(test_data_start, len(cleaned_original)):
        print(json.dumps(cleaned_indices[i]), file=f)

