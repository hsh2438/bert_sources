# bert based text classification with pytorch and transformers

## classification korean news sample
data: crawled 500 news

base model: bert multi-lingual model
https://drive.google.com/drive/folders/1aX8uxS8KwFspHjRYjSrE14bOhKSqU93z?usp=sharing

## make docker image
docker build . -t bert_classification

## train and evaluation
python classification.py

## start flask server
python server.py
