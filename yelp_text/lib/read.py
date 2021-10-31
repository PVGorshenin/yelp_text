import yaml

from yelp_text.lib.bert.load_bert import load_bert_model_n_tokenizer
from yelp_text.lib.bert.predict_bert import predict_bert
from yelp_text.definitions import get_root

def read_config() -> dict:
    with open(f'{get_root()}/config.yaml', 'r') as f:
        return yaml.load(f)


def _get_model_loaders():
    loaders_dct = {
        'bert': load_bert_model_n_tokenizer,
    }
    return loaders_dct


def _get_model_predictors():
    predictors_dct = {
        'bert': predict_bert
    }
    return predictors_dct


def load_models(config):
    models_dct = {}
    loaders_dct = _get_model_loaders()
    predict_func_dct = _get_model_predictors()
    for model_name in config['use_models']:
        model, preprocessor = loaders_dct[model_name](config)
        models_dct[model_name] = [model, preprocessor, predict_func_dct[model_name]]
    return models_dct
    
