import os
import pickle
from flask import Flask, request, jsonify
from flask_restplus import Api, Resource, fields

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import BertConfig, BertTokenizer, BertModel, DataProcessor, InputExample

from classification import ModelConfig, BertModelForClassification, predict


model_config = ModelConfig(model_path='out/pytorch_model.bin')
label_list = pickle.load(open(os.path.join(model_config.data_dir, 'labels.pickle'), 'rb'))
bert_config = BertConfig.from_json_file(model_config.bert_config)
tokenizer = BertTokenizer(model_config.vocab_file, do_lower_case=False)
model = BertModelForClassification.from_pretrained(model_config.model_path, config=bert_config, num_labels=len(label_list))
model.to(model_config.device)

def inference(title, content):
    example = InputExample(guid='predict-00', text_a = title, text_b = content, label = label_list[0])
    features = convert_examples_to_features([example], tokenizer, max_length=model_config.max_seq_length, label_list=label_list, output_mode="classification")
    
    result = predict(model_config, model, tokenizer, features)
    return result

_ = inference('test', 'init') # initializing


app = Flask(__name__)
api = Api(app, version='1.0', title='BERT classification')
ns  = api.namespace('classification') #namespace

classification_input = api.model('classification_input', {
    'title': fields.String(required=True, description='title', example='카카오톡 새해 벽두부터 2시간 넘게 먹통'),
    'content': fields.String(required=True, description='content', example='카카오톡이 경자년 시작과 함께 장애를 일으켜 새해 인사를 하려던 많은 이용자가 불편을 겪었다. 카카오에 따르면 1일 오전 0시부터 2시 15분까지 일부 사용자 카카오톡 메시지 수·발신이 원활하지 않은 현상이 발생했다. 카카오는 장애를 감지한 즉시 긴급 점검에 나서 시스템을 정상화했다. 회사측은 “카카오톡은 새해 인사 트래픽에 대비하는 비상 대응 모드를 매년 업그레이드하고 있다”며 “이번 연말을 대비해 새로 준비한 비상 대응 모드에서 예상하지 못한 시스템 오류가 발생해 폭증한 데이터를 원활하게 처리하지 못했다”고 밝혔다. 새해 시작과 함께 축하 메시지를 보내려는 이용자들은 카톡 먹통에 당혹해하며 사회관계망서비스 SNS 등을 통해 불편을 호소했다. 포털 사이트 실시간급상승검색어에 카카오톡 이 올라가기도 했다. 카카오 관계자는 “새해 첫날부터 불편을 겪으셨을 모든 분께 진심으로 사과의 말씀을 드린다”고 전했다.')
})

@ns.route('/')
class classification(Resource):
    
    @api.expect(classification_input)
    def post(self):
        title = request.json['title']
        content = request.json['content']
        predicted = inference(title, content)
        
        result = {}
        for idx, score in enumerate(predicted[0].tolist()):
            result[label_list[idx]] = score
        return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11122)
