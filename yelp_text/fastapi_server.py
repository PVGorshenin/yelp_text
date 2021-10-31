import json

from fastapi import FastAPI
from typing import Dict, List


def create_fastapi_server(predictor):
    app = FastAPI()

    @app.get('/')
    def handle_nothing():
        return 'server awaiting'

    @app.post("/")
    def handle_post(request: List[Dict[str: str]]):
        if len(request) != 0:
            preds_lst = predictor.predict(requests)
            # response_data = make_response_data(preds_lst)
        else:
            preds_lst = []

        return json.dumps(preds_lst, ensure_ascii=False).encode('utf-8')

    return app
