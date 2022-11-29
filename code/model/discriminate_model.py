import torch.nn as nn
import torch
from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
from transformers import OpenAIGPTConfig, OpenAIGPTModel, XLNetConfig, XLNetModel, DebertaV2Config, DebertaV2Model
from transformers import BartConfig, BartForSequenceClassification, GPT2Config, GPT2Model
from transformers import AutoConfig, AutoModel

# class prediction_layer(nn.Module):
#     def __init__(self):


class pretrained_model(nn.Module):
    def __init__(self, hps):
        super(pretrained_model, self).__init__()
        self.model_name = hps.model_name
        self.hps = hps
        if hps.model_name == 'bert' or hps.model_name == 'causalbert':
            self.model = BertModel.from_pretrained(hps.model_dir)
            self.config = BertConfig(hps.model_dir)
        elif hps.model_name == 'roberta':
            self.model = RobertaModel.from_pretrained(hps.model_dir)
            self.config = RobertaConfig(hps.model_dir)
        elif hps.model_name == 'albert':
            self.model = AlbertModel.from_pretrained(hps.model_dir)
            self.config = AlbertConfig.from_pretrained(hps.model_dir)
        elif hps.model_name == 'gpt':
            self.config = OpenAIGPTConfig.from_pretrained(hps.model_dir)
            self.model = OpenAIGPTModel.from_pretrained(hps.model_dir)
        elif hps.model_name == 'gpt2':
            self.config = GPT2Config.from_pretrained(hps.model_dir)
            self.model = GPT2Model.from_pretrained(hps.model_dir)
        elif hps.model_name == 'bart':
            self.config = BartConfig.from_pretrained(hps.model_dir)
            self.model = BartForSequenceClassification.from_pretrained(hps.model_dir)
        elif hps.model_name == 'xlnet':
            self.config = XLNetConfig.from_pretrained(hps.model_dir)
            self.model = XLNetModel.from_pretrained(hps.model_dir, mem_len=1024)
        elif hps.model_name == 'deberta':
            self.config = DebertaV2Config.from_pretrained(hps.model_dir)
            self.model = DebertaV2Model.from_pretrained(hps.model_dir)
        else: 
            self.model = AutoModel.from_pretrained(hps.model_dir)
            self.config = AutoConfig()

        if hps.loss_func == 'CrossEntropy':
            self.classification = nn.Linear(self.config.hidden_size, 1)
        else:
            self.classification = nn.Linear(self.config.hidden_size, 1)

        self.linear = nn.Linear(3, 1)

    def forward(self, input_ids, attention_mask, seg_ids=None, length=None):

        # model list: Bert, ALBERT, GPT
        if self.model_name in ['bert', 'albert', 'gpt', 'gpt2', 'causalbert']:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=seg_ids)
        elif self.model_name in ['deberta']:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=seg_ids, return_dict=False)
        # model list: Roberta, XLNet
        else:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # get the cls token for classification
        if self.model_name in ['bert', 'roberta', 'albert', 'causalbert']:
            cls_token = output[1]   # Bert, Roberta, ALBERT
        elif self.model_name in ['gpt']:
            print(f"{output.last_hidden_state.size()}\t{output[0].size()}")
            cls_token = output[0][range(output[0].shape[0]), length.cpu().tolist(), :]   # GPT
        elif self.model_name in ['gpt2']:
            print(f"{output.last_hidden_state.size()}\t{output[0].size()}")
            cls_token = output[0][range(output[0].shape[0]), length.cpu().tolist(), :]   # GPT-2
        elif self.model_name == 'deberta':
            cls_token = output[0][:, -1, :]  # DebertaV2Model
        elif self.model_name == 'xlnet':
            cls_token = output[0][:, -1, :]  # XLNet

        if self.model_name == 'bart':
            scores = self.linear(output[0])
        else:
            scores = self.classification(cls_token)

        return scores



