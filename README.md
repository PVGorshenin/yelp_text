# Local loading
1. `pip install .`
2. Change  `config.json[bert_params][device]` to `cpu`, if no GPU is available.
2. Load the server with `python main.py -p <PORT>`
3. Server will be available on `0.0.0.0:<PORT>`

# Запуск в Docker
1. Build an image
2. Change  `config.json[bert_params][device]` to `cpu`, if no GPU is available.
3. Run `docker run -p <PORT>:5000 -it [--gpus all] <IMAGE NAME>`
4. Run `python3 main.py -p 5000` inside container
5. Server will be available on port `<PORT>`

# Формат запросов
Input data format:
```
[
    'Some text 1',
    'Some text 2'
]
```

# Формат ответов

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
