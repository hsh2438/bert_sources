# convert bert model from tensorflow to pytorch

## requirement
tensorflow-cpu <br>
transformers

## convert
python convert_model_from_tf_to_pytorch.py --tf_checkpoint_path=model.ckpt --bert_config_file=bert_config.json --pytorch_dump_path=pytorch_model.bin 
