# bert based text classification
bert based text classification example <br>

## Information
data: crawled 500 korean news <br>
base model: bert multi-lingual model <br>
base model size is too big to upload github. you can download base model using followed link.
https://drive.google.com/drive/folders/1aX8uxS8KwFspHjRYjSrE14bOhKSqU93z?usp=sharing <br>

## Make docker image
docker build . -t bert_classification <br>

## Docker container start
docker run -it -p 11122:11122 --name classification bert_classification <br>

## Train and Evaluation
python classification.py <br>

## Start flask server with swagger
python server.py <br>

## Test
http://host:11122
