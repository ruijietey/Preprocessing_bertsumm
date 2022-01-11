# import glob
import math
from os import listdir
from os.path import isfile, join
from transformers import BertTokenizer
import multiprocessing
import torch
import time
import datetime

from utils.logging import init_logger, logger
from ext_sum import summarize
from models.model_builder import ExtSummarizer

# Configuration
MODEL_TYPE = 'distilbert'
MAX_SENT = 5 # 5 sentences extracted for CNN/DM datasets
INPUT_FP = "raw_data/less_stories/"
RESULT_FP = 'results/'
LOG_FP = 'logs/'
DATA_TYPE = "CNNDM"
PROC_COUNT = 5
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


def preprocess(chunk, model, tokenizer):
    for doc in chunk:
        start_time = time.time()
        logger.info("=============================")
        logger.info(f'Processing file: {doc} ...')
        input_fp = INPUT_FP + doc
        try:
            summarize(input_fp, RESULT_FP, model, MODEL_TYPE, tokenizer, max_length=MAX_SENT, data_type=DATA_TYPE)
        except:
            print(f"{input_fp} has issue with preprocessing summarization")
            logger.error(f"Has issue preprocessing {input_fp}")
        logger.info(f"Processing Time: {time.time() - start_time}s\n")
    pass
    # for i, doc in enumerate(documents):
    #     if doc[-5:] == "story" or doc[-3:] == "txt":
    #         start_time = time.time()
    #         logger.info(f'Document no. {i+1}')
    #         logger.info("=============================")
    #         logger.info(f'Processing file: {doc} ...')
    #         input_fp = INPUT_FP + doc
    #         summarize(input_fp, RESULT_FP, model, MODEL_TYPE, max_length=MAX_SENT, data_type=DATA_TYPE)
    #         logger.info(f"Processing Time: {time.time()-start_time}s\n=============================\n")
    #     else:
    #         raise IOError("Unknown file type")

def start_preprocess():
    # logger.info(f'CUDA:{torch.cuda.is_available()}')
    # gpu_ranks = [int(i) for i in range(len(VISIBLE_GPUS.split(',')))]
    # os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPUS
    preproc_time = time.time()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    init_logger(f'{LOG_FP+datetime.datetime.today().strftime("%d-%m-%Y")}.log')
    logger.info(f'Summarizing data from - "{INPUT_FP}" ...')
    logger.info(f'Maximum sentence: {MAX_SENT}. Data Type: {DATA_TYPE}')
    logger.info(f'Input: {INPUT_FP}. Output: {RESULT_FP}')
    # Load trained BertSUMExt model
    model = load_model()

    # Summarize and output to results for each doc
    documents = [f for f in listdir(INPUT_FP) if isfile(join(INPUT_FP, f))]
    chunk_size = math.floor(len(documents)/PROC_COUNT)
    logger.info(f"Chunk size: {chunk_size}")
    chunks = [documents[x:x + chunk_size] for x in range(0, len(documents), chunk_size)] # Break to smaller subsets

    # Preprocess files with multiprocessing
    procs = []
    for i in range(PROC_COUNT):
        p = multiprocessing.Process(target=preprocess, args=(chunks[i], model, tokenizer))
        # p = mp.Process(target=preprocess, args=(chunks[i], model, i))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    # p = multiprocessing.Pool(5)
    # for f in glob.glob(INPUT_FP + "*.story"):
    #     p.apply_async(preprocess, [f], model)

    # p.close()


        # print(len(result))

    # procs = []
    # for i in range(num_gpus):
    #     device_id = i
    #     procs.append(mp.Process(target=preprocess, args=(chunks, device_id, model), daemon=True))
    #     procs[i].start()
    #     logger.info(" Starting process pid: %d  " % procs[i].pid)
    # for p in procs:
    #     p.join()


if __name__ == "__main__":
    start_preprocess()