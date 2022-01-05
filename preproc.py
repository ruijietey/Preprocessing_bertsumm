import os
import torch
import nltk
from models.model_builder import ExtSummarizer
from ext_sum import summarize


# Configure model type to use (bert / distilbert)
MODEL_TYPE = 'distilbert'
MAX_SENT = 5 # 5 sentences extracted for CNN/DM datasets
INPUT_FP = "raw_data/example.story"
RESULT_FP = 'results/summary.txt'
DATA_TYPE = "CNN/DM"

def load_model():
    try:
        print(f"Loading {MODEL_TYPE} trained model...")
        # checkpoint = torch.load(f'./checkpoints/{model_type}.pt', map_location=lambda storage, loc: storage)
        checkpoint = torch.load(f'./checkpoints/{MODEL_TYPE}.pt', map_location="cpu")["model"]
        print(f"Model: {MODEL_TYPE} loaded.")
    except:
        raise IOError(f'checkpoint file does not exist - "./checkpoints/{MODEL_TYPE}.pt"')

    model = ExtSummarizer(device="cpu", checkpoint=checkpoint, bert_type=MODEL_TYPE)
    return model


def start_preprocess():
    # Load trained BertSUMExt model
    model = load_model()

    # Input
    # TODO: Read all stories to parse and save into jsonl of format - "text" and "summary" (gold summary)
    # TODO: Use BertSUMExt on the content to produce top sentences to produce index.jsonl (For CNN/DM, choose top 5 sentences)


    # Summarize
    print(f'Summarizing data from - "{INPUT_FP}" ...')
    summary = summarize(INPUT_FP, RESULT_FP, model, max_length=MAX_SENT, data_type=DATA_TYPE)
    print("Summary:")
    print(summary)


if __name__ == "__main__":
    start_preprocess()