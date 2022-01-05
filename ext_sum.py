import numpy as np
import torch
from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize
from utils import logger, init_logger
import json


DEVICE = 'cpu'

def preprocess(source_fp, data_type):
    """
    - Extract golden summary and original text from document
    - Remove \n
    - Sentence Tokenize
    - Add [SEP] [CLS] as sentence boundary
    """
    with open(source_fp) as source:
        raw_text = source.read().replace("\n", " ").replace("[CLS] [SEP]", " ")
    try:
        assert(data_type == "CNNDM")
        parts = raw_text.split("@highlight")
        raw_text = parts[0]
        summary = parts[1:]
        summary = [s.strip() + "." for s in summary]
        print(f'original: {raw_text}')
        print(f'gold summary: {summary}')
        sents = sent_tokenize(raw_text)
        processed_text = "[CLS] [SEP]".join(sents)
    except:
        raise NotImplementedError("Different type of data type used as input")

    return processed_text, summary, len(sents)


def load_text(processed_text, max_pos, device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    sep_vid = tokenizer.vocab["[SEP]"]
    cls_vid = tokenizer.vocab["[CLS]"]

    def _process_src(raw):
        raw = raw.strip().lower()
        raw = raw.replace("[cls]", "[CLS]").replace("[sep]", "[SEP]")
        src_subtokens = tokenizer.tokenize(raw)
        src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]
        src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        src_subtoken_idxs[-1] = sep_vid
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        
        segments_ids = []
        segs = segs[:max_pos]
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0
        return src, mask_src, segments_ids, clss, mask_cls

    src, mask_src, segments_ids, clss, mask_cls = _process_src(processed_text)
    segs = torch.tensor(segments_ids)[None, :].to(device)
    src_text = [[sent.replace("[SEP]", "").strip() for sent in processed_text.split("[CLS]")]]
    return src, mask_src, segs, clss, mask_cls, src_text


def get_selected_ids(model, input_data, result_path, max_length):
    with torch.no_grad():
        src, mask, segs, clss, mask_cls, src_str = input_data
        sent_scores, mask = model(src, segs, clss, mask, mask_cls)
        sent_scores = sent_scores + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)
        # print(f'src: {src}')
        print(f'src str: {src_str[0]}')
        print(f'selected ids: {selected_ids[0][:max_length]}')
        print(f'sentence scores: {sent_scores}')
        # for i, sid in enumerate(selected_ids[0][:max_length]):
        #     print(f'Sentence Ranking: {i+1} with ID: {sid}:')
        #     print(src_str[0][sid])

        return src_str[0], selected_ids[0][:max_length].tolist()


def summarize(raw_txt_fp, result_fp, model, model_type, max_length=3, max_pos=512, data_type="CNNDM"):
    main_data = {}
    index_data = {}
    model.eval()
    source_text, summary, full_length = preprocess(raw_txt_fp, data_type)
    input_data = load_text(source_text, max_pos, device=DEVICE)
    text, selected_ids = get_selected_ids(model, input_data, result_fp, max_length)   # Do not use block_trigram because Matchsum / Siamese-BERT will do semantic matching for at doc level

    # Output to JSONL
    main_data["text"] = text
    main_data["summary"] = summary
    index_data["sent_id"] = selected_ids
    main_fp = f'{result_fp}{data_type}_{model_type}.jsonl'
    index_fp = f'{result_fp}index.jsonl'
    with open(main_fp, 'a') as f:
        logger.info(json.dumps(main_data), file=f)
    with open(index_fp, 'a') as f:
        logger.info(json.dumps(index_data), file=f)
