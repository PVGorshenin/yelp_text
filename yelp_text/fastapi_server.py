import enum
import json
import uvicorn

from argparse import ArgumentParser
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from yelp_text.lib.preproc import pack_request_to_tuples_list
from yelp_text.lib.postproc import make_response_data



class Item(BaseModel):
    product_name: str
    product_description: str

    def __getitem__(self, key):
        return super().__getattribute__(key)

ListItem = List[Item]

def create_fastapi_server(predictor):
    app = FastAPI()

    @app.get('/')
    def handle_nothing():
        return 'server awaiting'

    @app.post("/")
    def handle_post(request: ListItem):
        if len(request) != 0:
            probability, preds_vote_ids, sure_mask = predictor.predict(
                pack_request_to_tuples_list(
                    request, ['product_name', 'product_description']
                )
            )
            response_data = make_response_data(probability,
                                      preds_vote_ids,
                                      sure_mask,
                                      )
        else:
            response_data = []

        return json.dumps(response_data, ensure_ascii=False).encode('utf-8')

    return app
