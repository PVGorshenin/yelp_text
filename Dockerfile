FROM nvidia/cuda:10.2-base

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3

RUN apt-get -y install python3-pip

COPY . /catalog_categorization
WORKDIR /catalog_categorization/

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r ./requirements.txt --no-cache-dir
RUN python3 -m nltk.downloader stopwords

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN cd /catalog_categorization/lib && python3 ./load_deeppavlov.py

CMD ["bash"]