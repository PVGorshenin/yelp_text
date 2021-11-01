FROM nvidia/cuda:10.2-base

RUN apt-get update && \
apt-get upgrade -y && \
apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install git


RUN python3 -m pip install --upgrade pip
RUN git clone https://github.com/PVGorshenin/yelp_text.git

WORKDIR yelp_text/
RUN python3 -m pip install -e .


RUN mkdir -p data/bert_model/
COPY data/bert_model/model_epoch6 data/bert_model/
WORKDIR yelp_text/


ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

CMD ["bash"]