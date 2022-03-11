import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartConfig
from transformers import BartTokenizer, modeling_bart
from rouge import Rouge


def tokenize_data(hps, data):
    tokenizer = BartTokenizer.from_pretrained(hps.model_dir)
    # tokenizer.pad_token = '<|endoftext|>'
    input_text = []
    labels = []
    truths = []
    for example in data:
        if example['ask-for'] == 'cause':
            input_text += [[example['alternative1'], example['premise']], [example['alternative2'], example['premise']]]
        else:
            input_text += [[example['premise'], example['alternative1']], [example['premise'], example['alternative2']]]
        labels += [1, 0] if example['label'] == 0 else [0, 1]
        truths += [example['general_truth']]*2
    input_tokenized = tokenizer(input_text, padding=True)
    input_ids = torch.LongTensor(input_tokenized['input_ids'])
    attention_mask = torch.LongTensor(input_tokenized['attention_mask'])

    output_tokenized = tokenizer(truths, padding=True)
    decoder_input_ids = torch.LongTensor(output_tokenized['input_ids'])
    decoder_attention_mask = torch.LongTensor(output_tokenized['attention_mask'])

    labels = torch.FloatTensor(labels)
    return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels


class discriminate_generate(nn.Module):
    def __init__(self, hps):
        super(discriminate_generate, self).__init__()
        self.hps = hps
        self.model = BartForConditionalGeneration.from_pretrained(hps.model_dir)
        self.config = BartConfig.from_pretrained(hps.model_dir)
        # self.linear = nn.Linear()
        self.linear1 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.linear2 = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_mask, labels, mode='train'):
        if mode == 'train':
            output = self.model(input_ids,
                                attention_mask,
                                decoder_input_ids=decoder_input_ids,
                                decoder_attention_mask=decoder_mask,
                                labels=decoder_input_ids)
            eos_mask = input_ids.eq(self.config.eos_token_id)
            sentence_representation = output[-1][eos_mask, :].view(output[-1].size(0), -1, output[-1].size(-1))[:, -1, :]
            # token_for_classification = output[2][:, 0, :]
            # dense = self.linear1(sentence_representation)
            score = self.linear2(sentence_representation).squeeze(1)
            gen_logits = output[1]
            return score, gen_logits
        else:
            output = self.model.generate(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         max_length=self.hps.length,
                                         # min_length=self.hps.length,
                                         # no_repeat_ngram_size=3,
                                         early_stopping=False,
                                         repetition_penalty=3,
                                         num_beams=30)
            return output


class generate_discriminate(nn.Module):
    def __init__(self, hps):
        super(generate_discriminate, self).__init__()
        self.hps = hps
        self.model = BartForConditionalGeneration.from_pretrained(hps.model_dir)
        self.config = BartConfig.from_pretrained(hps.model_dir)
        # self.linear = nn.Linear()
        self.linear1 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.linear2 = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_mask, labels, mode='train'):
        if mode == 'train':
            output = self.model(input_ids,
                                attention_mask,
                                decoder_input_ids=decoder_input_ids,
                                decoder_attention_mask=decoder_mask,
                                labels=decoder_input_ids,
                                output_hidden_states=True)
            eos_mask = decoder_input_ids.eq(self.config.eos_token_id)
            sentence_representation = output[2][-1][eos_mask, :].view(output[2][-1].size(0), -1, output[2][-1].size(-1))[:, -1, :]
            # token_for_classification = output[2][:, 0, :]
            # dense = self.linear1(sentence_representation)
            score = self.linear2(sentence_representation).squeeze(1)
            gen_logits = output[1]
            return score, gen_logits
        else:
            output = self.model.generate(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         max_length=self.hps.length,
                                         # min_length=self.hps.length,
                                         # no_repeat_ngram_size=3,
                                         early_stopping=False,
                                         repetition_penalty=3,
                                         num_beams=self.hps.beam_size)
            return output
















