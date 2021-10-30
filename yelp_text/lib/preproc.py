import re
from typing import List


def _clean_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ ]+', ' ', text.lower())
    text = text.replace('ё', 'е')
    return text


def get_lemma(text: str, morph, stopwords: list) -> str:
    text = _clean_text(text).split()
    new_text = []
    for word in text:
        p = morph.parse(word)[0]
        if p.normal_form not in stopwords:
            new_text.append(p.normal_form)
    result = " ".join(new_text)
    return result


def _join_if_2cols(text_list):
    return [' '.join(row) if isinstance(row, tuple) else row for row in text_list]


def stemm_tokenizer(text: str, stemmer, tokenizer) -> str:
    cleaned_text = _clean_text(text)
    stemmed_text = ' '.join([stemmer.stem(w) for w in tokenizer.tokenize(cleaned_text)])
    return stemmed_text


def get_vectors_lemm(text_list, vectorizer, morph, stopwords):
    text_list = _join_if_2cols(text_list)
    text_lst_lemm = [get_lemma(text, morph, stopwords) for text in text_list]
    X = vectorizer.transform(text_lst_lemm)
    return X


def get_vectors_stemm(text_list, vectorizer, stemmer, tokenizer):
    text_list = _join_if_2cols(text_list)
    text_lst_stemm = [stemm_tokenizer(text, stemmer, tokenizer) for text in text_list]
    X = vectorizer.transform(text_lst_stemm)
    return X


def pack_request_to_tuples_list(data: List[tuple], fields: list):
    return [(row[fields[0]], row[fields[1]]) for row in data]
