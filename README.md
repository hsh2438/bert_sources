# bert_sources
<br>
<br>

## convert bert model from tensorflow to pytorch

### requirement
tensorflow-cpu <br>
transformers

### convert
python convert_model_from_tf_to_pytorch.py --tf_checkpoint_path=model.ckpt --bert_config_file=bert_config.json --pytorch_dump_path=pytorch_model.bin 
<br>

## bert based text classification
bert based text classification example <br><br>

### Information
data: crawled 500 korean news (450 for train, 50 for evaluation) <br>
base model: bert multi-lingual model (base model is not included this repo because size of base model is too big to upload github. you can download base model using following link.)
https://drive.google.com/drive/folders/1aX8uxS8KwFspHjRYjSrE14bOhKSqU93z?usp=sharing <br><br>

### Make docker image
docker build . -t bert_classification <br><br>

### Docker container start
docker run -it -p 11122:11122 --name classification bert_classification <br><br>

### Train and Evaluation
python classification.py <br><br>

### Start flask server with swagger
python server.py <br><br>

### Test
http://host:11122
