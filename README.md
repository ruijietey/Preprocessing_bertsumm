# Preprocessing for MatchSUM

This repo is part of the pipeline for MatchSUM ([Zhong et al., 2020](https://github.com/maszhongming/MatchSum)). In order to do semantic matching, documents have to be preprocessed by extracting top sentences.

BERTSUM ([Liu et al., 2019](https://github.com/nlpyang/PreSumm)) is used for extracting these sentences and select top sentence ids, due to constraints of time and memory, DistilBERT ([Sanh et al., 2019](https://arxiv.org/abs/1910.01108)) a lite version of BERT is used.

Source codes from [PreSumm](https://github.com/nlpyang/PreSumm) is modified to use the HuggingFace's `transformers` library and their pretrained DistilBERT model. 

## References
- [1] [MatchSum: Extractive Summarization as Text Matching](https://github.com/maszhongming/MatchSum)
- [2] [PreSumm:  Text Summarization with Pretrained Encoders](https://github.com/nlpyang/PreSumm)
- [3] [DistilBERT: Smaller, faster, cheaper, lighter version of BERT](https://huggingface.co/transformers/model_doc/distilbert.html)
- [4] [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://github.com/google-research/google-research/tree/master/mobilebert)
- [5] [MobileBert_PyTorch](https://github.com/lonePatient/MobileBert_PyTorch)
