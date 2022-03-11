import torch.nn as nn
import torch
from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
from transformers import OpenAIGPTConfig, OpenAIGPTModel, XLNetConfig, XLNetModel
from transformers import BartConfig, BartForSequenceClassification

# class prediction_layer(nn.Module):
#     def __init__(self):


class pretrained_model(nn.Module):
    def __init__(self, hps):
        super(pretrained_model, self).__init__()
        self.model_name = hps.model_name
        self.hps = hps
        if hps.model_name == 'bert':
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
        elif hps.model_name == 'bart':
            self.config = BartConfig.from_pretrained(hps.model_dir)
            self.model = BartForSequenceClassification.from_pretrained(hps.model_dir)
        else:
            self.config = XLNetConfig.from_pretrained(hps.model_dir)
            self.model = XLNetModel.from_pretrained(hps.model_dir, mem_len=1024)

        if hps.loss_func == 'CrossEntropy':
            self.classification = nn.Linear(self.config.hidden_size, 2)
        else:
            self.classification = nn.Linear(self.config.hidden_size, 1)

        self.linear = nn.Linear(3, 1)

    def forward(self, input_ids, attention_mask, seg_ids=None, length=None):

        # model list: Bert, ALBERT, GPT
        if self.model_name in ['bert', 'albert', 'gpt']:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=seg_ids)

        # model list: Roberta, XLNet
        else:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # get the cls token for classification
        if self.model_name in ['bert', 'roberta', 'albert']:
            cls_token = output[1]   # Bert, Roberta, ALBERT
        elif self.model_name == 'gpt':
            cls_token = output[0][range(output[0].shape[0]), length.cpu().tolist(), :]   # GPT
        elif self.model_name == 'xlnet':
            cls_token = output[0][:, -1, :]  # XLNet

        if self.model_name == 'bart':
            scores = self.linear(output[0])
        else:
            scores = self.classification(cls_token)

        return scores



