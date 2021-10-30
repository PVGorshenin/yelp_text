import numpy as np
import pandas as pd
from typing import List


def _make_preds_df(preds_vote_ids):
    return pd.DataFrame({'le_class': preds_vote_ids})


def _make_merged_df(preds_df, probability, sure_mask, category_df):
    merged_df = preds_df.merge(category_df, left_on='le_class', right_on='le_class', how='inner')
    merged_df['probability'] = probability
    merged_df.loc[~sure_mask, 'probability'] = np.NaN
    return merged_df


def make_response_data(probability: np.ndarray, preds_vote_ids: np.ndarray,
                       sure_mask:np.ndarray, category_df: pd.DataFrame) -> List[dict]:
    preds_df = _make_preds_df(preds_vote_ids)
    merged_df = _make_merged_df(preds_df, probability, sure_mask, category_df)

    return merged_df[['parent_category_id', 'category_id', 'parent_category',
                      'category', 'probability']].to_dict('records')

