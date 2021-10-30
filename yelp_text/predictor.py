import yaml
from yelp_text.lib.mix_preds import mix_preds_borders_n_eqaulity


class Predictor():
    def __init__(self, models_dct):
        #TODO: abspath
        with open('../config.yaml', 'rb') as file:
            self.config = yaml.load(file)
        self.models_dct = models_dct

    def _get_model_aux_params(self):
        aux_params = {
            'bert': [self.config]
     #       'lr':  (self.stemmer, self.tokenizer, self.le)
        }
        return aux_params

    def predict(self, data):
        preds_lst, weights = [], []
        aux_params_dct = self._get_model_aux_params()
        for model_name in list(self.models_dct.keys()):
            model, preprocessor, predict_func = self.models_dct[model_name]

            aux_params = aux_params_dct[model_name]
            curr_preds = predict_func(data, model, preprocessor, *aux_params)

            preds_lst.append(curr_preds)
            weights.append(self.config[f'{model_name}_params']['weight'])

        probability, preds_vote_ids, sure_mask = mix_preds_borders_n_eqaulity(preds_lst, weights, self.config)
        return probability, preds_vote_ids, sure_mask

