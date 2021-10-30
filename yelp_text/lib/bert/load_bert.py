import torch

from definitions import get_root
from transformers import BertForSequenceClassification, BertTokenizer


def load_bert_model_n_tokenizer(config):
    device = config['bert_params']['device']
    model_path = config['bert_params']['model_path']

    state_dict = torch.load(f'{get_root()}/{model_path}')
    n_classes = next(reversed(state_dict.items()))[1].shape[0]
    #TODO: hardcode to config
    model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased',
                                                          state_dict=state_dict,
                                                          num_labels=n_classes)
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    return model, tokenizer

