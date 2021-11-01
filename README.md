# Before the run
I struggled to make auto download of model. And it did work locally, but not in the docker. So, please download 
model from here: https://drive.google.com/file/d/1Pxxa1-ViEZPARpp3rHGEQDXdEJs2ko_3/view?usp=sharing. And place them in:
`data/bert_model/model_epoch6` before docker build. 
Direct link, in case of download from code, could be found in `config[bert_params][model_url]`

Or you can pass it by **-v** flag. But it involves editing of Dockerfile.

# Local loading
1. `pip install .`
2. Change  `config[bert_params][device]` to `cpu`, if no GPU is available.
2. Load the server with `python main.py -p <PORT>`
3. Server will be available on `0.0.0.0:<PORT>`

#  Docker
1. Build an image
2. Change  `config[bert_params][device]` to `cpu`, if no GPU is available.
3. Run `docker run -p <PORT>:5000 -it [--gpus all] <IMAGE NAME>`
4. Run `python3 main.py -p 5000` inside container
5. Server will be available on port `<PORT>`

# Input format
Input data format:
```
[
    'Some text 1',
    'Some text 2'
]
```

# Output format

```
[
    {
        'text': input_text1,
        'n_useful_voting': float
    },
    {
        'text': input_text2,
        'n_useful_voting': float
    },
    ...
]
```
