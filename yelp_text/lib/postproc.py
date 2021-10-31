import numpy as np
from typing import List, Dict


def make_response_data(preds: np.ndarray, request: List[str]) -> List[Dict[str, float]]:
    res_lst = []
    for row in zip(request, preds.tolist()):
        res_lst.append({'text': row[0],
                        'n_useful_voting': row[1]})
    return res_lst

