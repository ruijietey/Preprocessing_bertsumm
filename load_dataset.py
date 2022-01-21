from datasets import load_dataset
from transformers import DistilBertTokenizer

raw_datasets = load_dataset('cnn_dailymail','3.0.0')
# print(raw_datasets)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def tokenize(data):
    return tokenizer(data['article'], data['highlights'], padding="max_length", truncation=True, max_length=512)


tokenized_datasets = raw_datasets.map(tokenize)
print(tokenized_datasets.column_names)
