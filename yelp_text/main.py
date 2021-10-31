import uvicorn
import logging

from argparse import ArgumentParser
from yelp_text.fastapi_server import create_fastapi_server
from yelp_text.predictor import Predictor
from yelp_text.lib.read import read_config, load_models

parser = ArgumentParser()
config = read_config()

models_dct = load_models(config)
predictor = Predictor(models_dct, config)

parser.add_argument('-p', '--port',
                    help='server port to listening',
                    type=int,
                    action='store',
                    dest='port')
args = parser.parse_args()


app = create_fastapi_server(predictor)

logging.warning(f'Server started on 0.0.0.0:{args.port}\n')
uvicorn.run(app, host="0.0.0.0", port=args.port)
logging.warning('Server stopped\n')