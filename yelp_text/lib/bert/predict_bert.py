import torch
import numpy as np

from typing import Union, List, Tuple
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import softmax


def _get_preds(model, dataloader, device: str) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)

            inputs = {
                'input_ids': batch[0],
                'token_type_ids': batch[1],
                'attention_mask': batch[2],
            }

            logits = model(**inputs)['logits']
            logits = logits.detach().cpu().numpy()

            preds.append(logits)

    preds = np.concatenate(preds, axis=0)
    return preds


def predict_bert(text_or_pair_list: Union[List[str], List[Tuple[str, str]]],
                 model,
                 tokenizer,
                 config: str) -> np.ndarray:
    params = config['bert_params']

    encoded_data = tokenizer.batch_encode_plus(
        text_or_pair_list,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=params['padding'],
        truncation=params['truncation'],
        max_length=params['max_seq_len'],
        return_tensors='pt'
    )

    dataset = TensorDataset(encoded_data['input_ids'],
                            encoded_data['token_type_ids'],
                            encoded_data['attention_mask'])

    dataloader = DataLoader(dataset,
                            batch_size=params['batch_size'],
                            shuffle=False)

    preds_np = _get_preds(model, dataloader, params['device'])

    return preds_np