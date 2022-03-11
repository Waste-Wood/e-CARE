import torch
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import BartConfig, BartModel, BartForConditionalGeneration
from rouge import Rouge
import torch.nn as nn


class gpt2_generate(nn.Module):
    def __init__(self, hps):
        super(gpt2_generate, self).__init__()
        self.hps = hps
        self.model_dir = hps.model_dir

        self.config = GPT2Config.from_pretrained(self.model_dir)
        # self.model = GPT2Model.from_pretrained(self.model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_dir)
        # self.output_vocab = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None, past_key_values=None, mode='train', true_labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=true_labels, past_key_values=past_key_values)
        if mode == 'train':
            return outputs[0]
        else:
            return outputs







