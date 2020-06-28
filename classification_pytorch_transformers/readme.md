# bert based text classification with pytorch and transformers
<br>
## classification korean news sample
data: crawled 500 korean news
base model: bert multi-lingual model <br>
https://drive.google.com/drive/folders/1aX8uxS8KwFspHjRYjSrE14bOhKSqU93z?usp=sharing

## make docker image
docker build . -t bert_classification <br>

## docker container start
docker run -it -p 11122:11122 --name classification bert_classification <br>

## train and evaluation
python classification.py <br>

## start flask server
python server.py <br>
