from os import listdir
from os.path import isfile, join
import torch
import time
from ext_sum import summarize
from models.model_builder import ExtSummarizer

# Configuration
MODEL_TYPE = 'distilbert'
MAX_SENT = 5 # 5 sentences extracted for CNN/DM datasets
INPUT_FP = "raw_data/less_stories/"
RESULT_FP = 'results/'
DATA_TYPE = "CNNDM"
DEVICE = "cpu"


def load_model():
    try:
        print(f"Loading {MODEL_TYPE} trained model...")
        # checkpoint = torch.load(f'./checkpoints/{model_type}.pt', map_location=lambda storage, loc: storage)
        checkpoint = torch.load(f'./checkpoints/{MODEL_TYPE}.pt', map_location=DEVICE)["model"]
        print(f"Model: {MODEL_TYPE} loaded.")
    except:
        raise SystemError(f'checkpoint file does not exist OR invalid device - "./checkpoints/{MODEL_TYPE}.pt"')

    model = ExtSummarizer(device=DEVICE, checkpoint=checkpoint, bert_type=MODEL_TYPE)
    return model


def start_preprocess():
    print(f'Summarizing data from - "{INPUT_FP}" ...')
    print(f'Maximum sentence: {MAX_SENT}. Data Type: {DATA_TYPE}')
    print(f'Input: {INPUT_FP}. Output: {RESULT_FP}')
    # Load trained BertSUMExt model
    model = load_model()

    # Summarize and output to results for each doc
    documents = [f for f in listdir(INPUT_FP) if isfile(join(INPUT_FP, f))]
    for i, doc in enumerate(documents):
        if doc[-5:] == "story" or doc[-3:] == "txt":
            start_time = time.time()
            print(f'Document no. {i+1}')
            print("=============================")
            print(f'Processing file: {doc} ...')
            input_fp = INPUT_FP + doc
            summarize(input_fp, RESULT_FP, model, MODEL_TYPE, max_length=MAX_SENT, data_type=DATA_TYPE)
            print(f"Processing Time: {time.time()-start_time}s\n=============================\n")
        else:
            raise IOError("Unknown file type")


if __name__ == "__main__":
    start_preprocess()