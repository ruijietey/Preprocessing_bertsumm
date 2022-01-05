import os
import torch
import nltk
from models.model_builder import ExtSummarizer
from ext_sum import summarize


# Configure model type to use (bert / distilbert)
MODEL_TYPE = 'distilbert'
MAX_SENT = 5 # 5 sentences extracted for CNN/DM datasets
INPUT_FP = "raw_data/example.story"
RESULT_FP = 'results/'
DATA_TYPE = "CNNDM"


def start_preprocess():
    # Load trained BertSUMExt model

    # Summarize
    print(f'Summarizing data from - "{INPUT_FP}" ...')
    summarize(INPUT_FP, RESULT_FP, MODEL_TYPE, max_length=MAX_SENT, data_type=DATA_TYPE)


if __name__ == "__main__":
    start_preprocess()