import json

from fastapi import FastAPI
from typing import  List
from yelp_text.lib.postproc import make_response_data


def create_fastapi_server(predictor):
    app = FastAPI()

    @app.get('/')
    def handle_nothing():
        return 'server awaiting'

    @app.post("/")
    def handle_post(request: List[str]):
        if len(request) != 0:
            preds_lst = predictor.predict(request)
            #TODO: generalize for n models
            response_data = make_response_data(preds_lst[0], request)
        else:
            response_data = []

        return json.dumps(response_data, ensure_ascii=False).encode('utf-8')

    return app
