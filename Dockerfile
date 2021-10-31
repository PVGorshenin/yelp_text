FROM nvidia/cuda:10.2-base

RUN apt-get update && \
apt-get upgrade -y && \
apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install git


RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install 'git+https://github.com/PVGorshenin/yelp_text@main'

RUN gdown 'https://drive.google.com/uc?export=download&id=1Pxxa1-ViEZPARpp3rHGEQDXdEJs2ko_3'
RUN mkdir data/bert_model/ && mv model_epoch6 data/bert_model/

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

CMD ["bash"]