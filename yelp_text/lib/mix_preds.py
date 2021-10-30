import numpy as np
from typing import List, Union, Tuple

        
def vote_preds(pred_lst, koeff_lst):
    preds_vote = np.zeros_like(pred_lst[0])
    assert len(pred_lst) == len(koeff_lst)
    assert np.isclose(np.sum(koeff_lst), 1)
    for i_pred, i_koeff in zip(pred_lst, koeff_lst):
        preds_vote += i_pred * i_koeff
    return preds_vote


def is_equal_choice(pred_lst):
    is_equal_choice = [True] * pred_lst[0].shape[0]
    if len(pred_lst)>1:
        pred_ids = [np.argmax(preds, axis=1) for preds in pred_lst]
        for i_pred in range(len(pred_ids)-1):
            is_equal_choice &= (pred_ids[i_pred] == pred_ids[i_pred+1])
    return is_equal_choice


def mix_preds_borders_n_eqaulity(preds_lst: List[np.ndarray], weights: list,
                                 config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get predictions with probability boreder and equality of models choices.
    
    Get n-models mixture of predictions with check if all using models give equal predictions
    and if all predictions have higher probability than target level.

    """
    
    preds_vote = vote_preds(preds_lst, weights)
    preds_vote_ids = np.argmax(preds_vote, axis=1)
    preds_vote_proba = np.max(preds_vote, axis=1)
    sure_mask = [True] * len(preds_lst[0])
    if config['confidence_level'] is not None: 
        is_above_lvl = preds_vote_proba > float(config['confidence_level'])
        sure_mask &= is_above_lvl
    if config['is_equal_choice']:
        is_equal_choice_mask = is_equal_choice(preds_lst)
        sure_mask &= is_equal_choice_mask
    return preds_vote_proba, preds_vote_ids, sure_mask


