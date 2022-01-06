# import glob
import torch
import datetime

from utils.logging import init_logger, logger
from ext_sum import preprocess_multi
from models.model_builder import ExtSummarizer

# Configuration
MODEL_TYPE = 'distilbert'
MAX_SENT = 5 # 5 sentences extracted for CNN/DM datasets
INPUT_FP = "raw_data/less_stories/"
RESULT_FP = 'results/'
LOG_FP = 'logs/'
DATA_TYPE = "CNNDM"
DEVICE = 'cpu'
# DEVICE = "cuda"
# VISIBLE_GPUS = "0,1,2,3"


def load_model():
    try:
        logger.info(f"Loading {MODEL_TYPE} trained model...")
        # checkpoint = torch.load(f'./checkpoints/{MODEL_TYPE}.pt', map_location=lambda storage, loc: storage)["model"]
        checkpoint = torch.load(f'./checkpoints/{MODEL_TYPE}.pt', map_location=DEVICE)["model"]
        logger.info(f"Model: {MODEL_TYPE} loaded.")
    except:
        raise SystemError(f'checkpoint file does not exist OR invalid device - "./checkpoints/{MODEL_TYPE}.pt"')

    model = ExtSummarizer(device=DEVICE, checkpoint=checkpoint, bert_type=MODEL_TYPE).to(DEVICE)
    return model


def start_preprocess():
    init_logger(f'{LOG_FP+datetime.datetime.today().strftime("%d-%m-%Y")}.log')
    logger.info(f'Summarizing data from - "{INPUT_FP}" ...')
    logger.info(f'Maximum sentence: {MAX_SENT}. Data Type: {DATA_TYPE}')
    logger.info(f'Input: {INPUT_FP}. Output: {RESULT_FP}')
    # Load trained BertSUMExt model
    model = load_model()

    # Summarize and output to results for each doc

    preprocess_multi(INPUT_FP, RESULT_FP, model, MODEL_TYPE, MAX_SENT, DATA_TYPE)

if __name__ == "__main__":
    start_preprocess()