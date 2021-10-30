import dill
import numpy as np

from definitions import get_root
from yelp_text.lib.preproc import get_vectors_stemm
from typing import List, Tuple, Union


def load_lr_model_n_vect(config: dict) -> tuple:
    vectorizer_path = config['lr_params']['vectorizer_path']
    lr_path = config['lr_params']['model_path']

    with open(f'{get_root()}/{vectorizer_path}', 'rb') as f:
        vectorizer = dill.load(f)

    with open(f'{get_root()}/{lr_path}', 'rb') as f:
        lr_model = dill.load(f)

    return lr_model, vectorizer


def predict_lr(text_lst:  Union[List[str], List[Tuple[str, str]]],
               lr_model, vectorizer, stemmer, tokenizer, label_encoder) -> np.ndarray:
    all_possible_classes = label_encoder.classes_
    model_saw_classes = lr_model.classes_

    preds = np.zeros((len(text_lst), len(all_possible_classes)))
    X = get_vectors_stemm(text_lst, vectorizer, stemmer, tokenizer)
    preds[:, model_saw_classes] = lr_model.predict_proba(X)
    return preds
