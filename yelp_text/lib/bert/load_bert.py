import torch

from yelp_text.definitions import get_root
from transformers import BertForSequenceClassification, BertTokenizer


def load_bert_model_n_tokenizer(config):
    device = config['bert_params']['device']
    model_path = config['bert_params']['model_path']
    model_type = config['bert_params']["model_type"]
    root = get_root()

    if not torch.cuda.is_available():
        state_dict = torch.load(f'{root}/{model_path}', map_location=device)
    else:
        state_dict = torch.load(f'{root}/{model_path}')

    model = BertForSequenceClassification.from_pretrained(model_type,
                                                          state_dict=state_dict,
                                                          num_labels=1)

    tokenizer = BertTokenizer.from_pretrained(model_type)
    return model, tokenizer
