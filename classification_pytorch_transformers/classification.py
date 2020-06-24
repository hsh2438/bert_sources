import os
import logging
import random
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

from transformers import BertConfig, BertTokenizer, BertPreTrainedModel, BertModel, DataProcessor, InputExample
from transformers import glue_convert_examples_to_features as convert_examples_to_features


logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(self, \
            bert_config = 'model/bert_config.json', \
            model_path = 'model/pytorch_model.bin', \
            vocab_file = 'model/vocab.txt', \
            data_dir = 'data', \
            output_dir = 'out', \
            epoch = 10, \
            learning_rate = 2e-5, \
            batch_size = 8):

        self.bert_config = bert_config
        self.model_path = model_path
        self.vocab_file = vocab_file

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_seq_length = 512

        self.batch_size = batch_size
        self.num_gpu = 1
        self.epoch = epoch
        self.learning_rate = learning_rate
        
        self.seed = 42
        self.gradient_accumulation_steps = 1
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.max_grad_norm = 1.0

        self.device = 'cuda:{}'.format(self.num_gpu) if self.num_gpu > -1 else 'cpu'


class BertModelForClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=3):
        super().__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class ClassificationProcessor(DataProcessor):
    """Processor for the dataset."""        

    def get_train_examples(self, data_dir):
        """See base class."""
        labels = set()
        lines = self._read_tsv(os.path.join(data_dir, 'train.tsv'))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[3]
            text_b = line[4]
            
            label = line[1]
            labels.add(label)
            
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        pickle.dump(sorted(list(labels)), open(os.path.join(data_dir, 'labels.pickle'), 'wb'))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)
            text_a = line[3]
            text_b = line[4]
            
            label = line[1]
            
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        return pickle.load(open(os.path.join(data_dir, 'labels.pickle'), 'rb'))


def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.num_gpu > -1:
        torch.cuda.manual_seed_all(config.seed)


def load_and_cache_examples(config, tokenizer, evaluate=False):

    processor = ClassificationProcessor()

    logger.info("Creating features from dataset file at %s", config.data_dir)
    examples = (
        processor.get_test_examples(config.data_dir) if evaluate else processor.get_train_examples(config.data_dir)
    )
    label_list = processor.get_labels(config.data_dir)
    features = convert_examples_to_features(
        examples, tokenizer, max_length=config.max_seq_length, label_list=label_list, output_mode="classification",
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def train(config, model, tokenizer):
    """ Train the model """
    train_dataset = load_and_cache_examples(config, tokenizer, evaluate=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size)

    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.epoch

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.epoch)
    logger.info("  Instantaneous batch size = %d", config.batch_size)
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(config.epoch), desc="Epoch"
    )
    set_seed(config)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(config.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "token_type_ids": batch[2]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

    os.mkdir(config.output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(config.output_dir)

    return global_step, tr_loss / global_step


def evaluate(config, model, tokenizer, prefix=""):

    eval_dataset = load_and_cache_examples(config, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", config.batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(config.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2], "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    
    result = (preds == out_label_ids).mean()

    return result


if __name__ == '__main__':
    
    model_config = ModelConfig(epoch=2, learning_rate=2e-5, batch_size=4)

    bert_config = BertConfig.from_json_file(model_config.bert_config)
    tokenizer = BertTokenizer(model_config.vocab_file, do_lower_case=False)
    model = BertModelForClassification.from_pretrained(model_config.model_path, config=bert_config, num_labels=5)
    model.to(model_config.device)

    global_step, tr_loss = train(model_config, model, tokenizer)

    eval_result = evaluate(model_config, model, tokenizer)
    print('eval_result: ', eval_result)
