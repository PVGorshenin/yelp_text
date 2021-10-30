import numpy as np

def is_equal_choice(pred_lst):
    is_equal_choice = [True] * pred_lst[0].shape[0]
    if len(pred_lst)>1:
        pred_ids = [np.argmax(preds, axis=1) for preds in pred_lst]
        for i_pred in range(len(pred_ids)-1):
            is_equal_choice &= (pred_ids[i_pred] == pred_ids[i_pred+1])
    return is_equal_choice
