import pickle
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, OpenAIGPTTokenizer, XLNetTokenizer
from transformers import GPT2Tokenizer, BartTokenizer
import torch
import logging
import sys
from rouge import Rouge
from nltk import bleu
from tqdm import trange
import torch.nn.functional as F
from tqdm import trange
from nlp import load_dataset
from tqdm import tqdm
import datetime
import csv
import pdb
import torch.nn as nn
import json


def tokenize_data(data, model_path, model_name):
    # tokenizer = BertTokenizer(vocab_file=model_path+'/'+'vocab.txt')
    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(model_path)
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(model_path)

    # unique ids
    cls_id = tokenizer._convert_token_to_id('[CLS]')
    sep_id = tokenizer._convert_token_to_id('[SEP]')
    pad_id = tokenizer._convert_token_to_id('[PAD]')

    labels = []
    instances = []
    segments = []

    max_length = 0

    # tokenization
    for example in data:
        premise, a1, a2 = example['premise'], example['alternative1'], example['alternative2']
        premise_id = tokenizer.convert_tokens_to_ids(tokenizer._tokenize(premise))
        a1_id = tokenizer.convert_tokens_to_ids(tokenizer._tokenize(a1))
        a2_id = tokenizer.convert_tokens_to_ids(tokenizer._tokenize(a2))
        max_length = max(max_length, len(premise_id + a1_id) + 3, len(premise_id + a2_id) + 3)
        if example['ask-for'] == 'cause':
            instance1 = [cls_id] + a1_id + [sep_id] + premise_id + [sep_id]
            seg1 = [0] * (len(a1_id) + 2) + [1] * (len(premise_id) + 1)
            instance2 = [cls_id] + a2_id + [sep_id] + premise_id + [sep_id]
            seg2 = [0] * (len(a2_id) + 2) + [1] * (len(premise_id) + 1)
        else:
            instance1 = [cls_id] + premise_id + [sep_id] + a1_id + [sep_id]
            seg1 = [0] * (len(premise_id) + 2) + [1] * (len(a1_id) + 1)
            instance2 = [cls_id] + premise_id + [sep_id] + a2_id + [sep_id]
            seg2 = [0] * (len(premise_id) + 2) + [1] * (len(a2_id) + 1)
        instances += [instance1, instance2]
        segments += [seg1, seg2]
        labels += [0, 1] if example['label'] == 1 else [1, 0]

    # padding
    segments = [seg + [0] * (max_length - len(seg)) for seg in segments]
    attention_mask = [[1] * len(instance) + [0] * (max_length - len(instance)) for instance in instances]
    instances = [instance + [pad_id] * (max_length - len(instance)) for instance in instances]

    return torch.LongTensor(instances), torch.LongTensor(attention_mask), torch.LongTensor(segments), torch.LongTensor(labels)


def tokenize_multi_choices(data, hps):
    # load pretrained tokenizer
    if hps.model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(hps.model_dir)
    elif hps.model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(hps.model_dir)
    elif hps.model_name == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(hps.model_dir)
    elif hps.model_name == 'gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(hps.model_dir, unk_token="<unk>")
        tokenizer.pad_token = tokenizer.unk_token
    elif hps.model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)
        tokenizer.pad_token = tokenizer.unk_token
    elif hps.model_name == 'bart':
        tokenizer = BartTokenizer.from_pretrained(hps.model_dir)
    else:
        tokenizer = XLNetTokenizer.from_pretrained(hps.model_dir)
    
    instances = []
    labels = []

    # pdb.set_trace()
    for example in data:
        if hps.data_name == 'because' or hps.data_name == 'event_storyline':
            premise, hypothesis = example['premise'], example['hypothesis']
            instance = [premise, hypothesis]
            labels.append(example['label'])
            instances.append(instance)
        elif hps.data_name == 'commonsenseqa':
            premise, alternatives = example['premise'], example['alternatives']
            label = example['label']
            tmp_instances = [[premise, alternative] for alternative in alternatives]
            tmp_labels = [0 for _ in range(len(alternatives))]
            tmp_labels[label] = 1
            labels += tmp_labels
            instances += tmp_instances
    
    outputs = tokenizer(instances, padding=True, return_token_type_ids=True, return_length=True)
    input_ids = outputs['input_ids']
    attention_mask = outputs['attention_mask']
    token_type_ids = outputs['token_type_ids']
    length = outputs['length']

    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), \
           torch.LongTensor(token_type_ids), torch.LongTensor(labels), torch.LongTensor(length)-1


def quick_tokenize(data, hps):
    # load pretrained tokenizer
    if hps.model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(hps.model_dir)
    elif hps.model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(hps.model_dir)
    elif hps.model_name == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(hps.model_dir)
    elif hps.model_name == 'gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(hps.model_dir, unk_token="<unk>")
        tokenizer.pad_token = tokenizer.unk_token
    elif hps.model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)
        tokenizer.pad_token = tokenizer.unk_token
    elif hps.model_name == 'bart':
        tokenizer = BartTokenizer.from_pretrained(hps.model_dir)
    else:
        tokenizer = XLNetTokenizer.from_pretrained(hps.model_dir)

    instances = []
    labels = []
    for example in data:
        premise, a1, a2 = example['premise'], example['alternative1'], example['alternative2']

        if example['ask-for'] == 'cause':
            if not hps.hyp_only:
                instance1 = [a1, premise]
                instance2 = [a2, premise]
            else:
                instance1 = a1
                instance2 = a2
        else:
            if not hps.hyp_only:
                instance1 = [premise, a1]
                instance2 = [premise, a2]
            else:
                instance1 = a1
                instance2 = a2
        labels += [0, 1] if example['label'] == 1 else [1, 0]
        instances += [instance1, instance2]

    outputs = tokenizer(instances, padding=True, return_token_type_ids=True, return_length=True)
    input_ids = outputs['input_ids']
    attention_mask = outputs['attention_mask']
    token_type_ids = outputs['token_type_ids']
    length = outputs['length']

    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), \
           torch.LongTensor(token_type_ids), torch.LongTensor(labels), torch.LongTensor(length)-1


def tokenize_multi_task(hps, data):
    tokenizer = RobertaTokenizer.from_pretrained(hps.discriminate_model_dir)
    instances1 = []
    instances2 = []
    labels = []
    truths = []

    for example in data:
        truth, premise, a1, a2 = example['general_truth'], example['premise'], example['alternative1'], example[
            'alternative2']
        truths.append(truth)
        if example['ask-for'] == 'cause':
            instances1.append([a1, premise])
            instances2.append([a2, premise])
        else:
            instances1.append([premise, a1])
            instances2.append([premise, a2])
        labels += [example['label']]

    outputs1 = tokenizer(instances1, padding=True)
    outputs2 = tokenizer(instances2, padding=True)
    outputs_truth = tokenizer(truths, padding=True)
    input_ids1 = torch.LongTensor(outputs1['input_ids'])
    input_ids2 = torch.LongTensor(outputs2['input_ids'])
    truth_ids = torch.LongTensor(outputs_truth['input_ids'])
    mask1 = torch.LongTensor(outputs1['attention_mask'])
    mask2 = torch.LongTensor(outputs2['attention_mask'])
    mask_truth = torch.LongTensor(outputs_truth['attention_mask'])

    return input_ids1, input_ids2, truth_ids[:, 1:], mask1, mask2, mask_truth[:, 1:], torch.LongTensor(labels)


def compute_ppl(hps, model, data):
    # device = 'cuda'
    if hps.model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)
        lls = []
        total_length = 0
        for example in data:
            input_text = example['cause'] + ' ' + example['effect']
            truth = example['general_truth']
            inputs = tokenizer(input_text)
            input_ids = torch.LongTensor(inputs['input_ids']).unsqueeze(0).cuda()
            attention_mask = torch.LongTensor(inputs['attention_mask']).unsqueeze(0).cuda()
            label_inputs = tokenizer(truth)
            label_ids = torch.LongTensor(label_inputs['input_ids']).unsqueeze(0).cuda()
            length = label_ids.shape[1]
            total_length += length
            
            # label_mask = torch.LongTensor(label_inputs['attention_mask']).unsqueeze(0).cuda()
            attention_mask = torch.cat((attention_mask, torch.ones(1, label_ids.shape[1]).long().cuda()), 1)
            label_ids = torch.cat((torch.LongTensor([-100]*input_ids.shape[1]).unsqueeze(0).cuda(), label_ids), 1)
            input_ids = torch.cat((input_ids, label_ids[:, input_ids.shape[1]:]), 1)
            with torch.no_grad():
                loss = model(input_ids, attention_mask=attention_mask, labels=label_ids)[0]
                lls.append(loss * length)
                
        ppl = torch.exp(torch.stack(lls).sum() / total_length)

    else:
        tokenizer = BartTokenizer.from_pretrained(hps.model_dir)
        lls = []
        total_length = 0
        for example in data:
            input_text = example['cause'] + ' ' + example['effect']
            truth = example['general_truth']
            inputs = tokenizer(input_text)
            input_ids = torch.LongTensor(inputs['input_ids']).unsqueeze(0).cuda()
            attention_mask = torch.LongTensor(inputs['attention_mask']).unsqueeze(0).cuda()
            label_inputs = tokenizer(truth)
            label_ids = torch.LongTensor(label_inputs['input_ids']).unsqueeze(0).cuda()
            length = label_ids.shape[1]
            total_length += length
            label_mask = torch.LongTensor(label_inputs['attention_mask']).unsqueeze(0).cuda()
            # attention_mask = torch.cat((attention_mask, torch.ones(1, label_ids.shape[1]).long().cuda()), 1)
            # label_ids = torch.cat((torch.LongTensor([-100]*input_ids.shape[1]).unsqueeze(0).cuda(), label_ids), 1)
            # input_ids = torch.cat((input_ids, label_ids[:, input_ids.shape[1]:]), 1)
            with torch.no_grad():
                loss = model(input_ids, attention_mask=attention_mask, decoder_input_ids=label_ids, decoder_attention_mask=label_mask, labels=label_ids)[0]
                lls.append(loss * length)

        ppl = torch.exp(torch.stack(lls).sum() / total_length)

    return ppl.item()


def evaluate_multi_task(model, dataloader_input, dataloader_output, hps):
    tokenizer = BartTokenizer.from_pretrained(hps.model_dir)
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    count = 0
    for batch1, batch2, t in zip(dataloader_input, dataloader_output, trange(len(dataloader_input))):
        if hps.cuda:
            batch1 = tuple(term.cuda() for term in batch1)
            batch2 = tuple(term.cuda() for term in batch2)

        input_ids, attention_mask, labels = batch1
        decoder_ids, decoder_mask = batch2
        scores, _ = model(input_ids,
                          attention_mask,
                          decoder_ids,
                          decoder_mask,
                          labels,
                          mode='train')
        scores = torch.cat((scores[::2].unsqueeze(1), scores[1::2].unsqueeze(1)), 1)
        index = torch.argmax(scores, 1)
        predict_labels = index.cpu().tolist()
        labels = torch.cat((labels[::2].unsqueeze(1), labels[1::2].unsqueeze(1)), 1)
        labels = torch.argmax(labels, 1).cpu().tolist()
        for k in range(len(predict_labels)):
            if labels[k] == predict_labels[k]:
                count += 1
            else:
                continue

        input_ids = torch.cat((input_ids[::2].unsqueeze(1), input_ids[1::2].unsqueeze(1)), 1)
        input_ids = input_ids[range(input_ids.shape[0]), index, :]
        attention_mask = torch.cat((attention_mask[::2].unsqueeze(1), attention_mask[1::2].unsqueeze(1)), 1)
        attention_mask = attention_mask[range(attention_mask.shape[0]), index, :]

        # for i in range(input_ids.shape[0]):
        gen_ids = model(input_ids,
                        attention_mask,
                        decoder_ids,
                        decoder_mask,
                        labels,
                        mode='generate')
        generated_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in gen_ids.tolist()]
        gold_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in decoder_ids[::2, :].tolist()]

        for i in range(len(generated_text)):
            bleu1 += bleu([gold_text[i]], generated_text[i], [1, 0, 0, 0])
            bleu2 += bleu([gold_text[i]], generated_text[i], [0, 1, 0, 0])
            bleu3 += bleu([gold_text[i]], generated_text[i], [0, 0, 1, 0])
            bleu4 += bleu([gold_text[i]], generated_text[i], [0, 0, 0, 1])

    num_instances = (len(dataloader_output) - 1) * hps.batch_size // 2 + input_ids.shape[0]
    return count / num_instances, bleu1 / num_instances, bleu2 / num_instances, bleu3 / num_instances, bleu4 / num_instances


# def evaluate_multi_task(model, dataloader, hps):
#     tokenizer = GPT2Tokenizer.from_pretrained(hps.generate_model_dir)
#     bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
#     count = 0
#     for batch in dataloader:
#         if hps.cuda:
#             batch = tuple(term.cuda() for term in batch)

#         ids1, ids2, ids3, mask1, mask2, mask3, label = batch
#         probs, _, gen_model, sentences, attention_mask = model(ids1, mask1, ids2, mask2, ids3, mask3)
#         predict_label = torch.argmax(probs, 1)
#         count += torch.sum(predict_label == label).item()

#         output = sample_sequence(gen_model, hps.length, device='cuda', context=sentences, batch_size=hps.batch_size,
#                                  attention_mask=attention_mask, input_type='embeddings')

#         generated = output

#         for i in range(generated.shape[0]):
#             predict_tokens = tokenizer.convert_ids_to_tokens(generated[i])
#             generated_text = remove_special_tokens(tokenizer.convert_tokens_to_string(predict_tokens))

#             gold_tokens = tokenizer.convert_ids_to_tokens(ids3[i])
#             gold_text = remove_special_tokens(tokenizer.convert_tokens_to_string(gold_tokens))

#             bleu1 += bleu([gold_text], generated_text, [1, 0, 0, 0])
#             bleu2 += bleu([gold_text], generated_text, [0, 1, 0, 0])
#             bleu3 += bleu([gold_text], generated_text, [0, 0, 1, 0])
#             bleu4 += bleu([gold_text], generated_text, [0, 0, 0, 1])

#     num_instances = (len(dataloader) - 1) * hps.batch_size + ids1.shape[0]
#     return count / num_instances, bleu1 / num_instances, bleu2 / num_instances, bleu3 / num_instances, bleu4 / num_instances



def load_data(path):
    data = [json.laods(line) for line in open(path, 'r')]
    return data


def evaluation(hps, dataloader, model, loss_function, mode='train'):
    predictions = []
    labels = []
    loss = 0
    model.eval()
    for batch in dataloader:
        if hps.cuda:
            batch = tuple(term.cuda() for term in batch)

        if mode == 'train':
            sent, seg_id, atten_mask, tmp_labels, tmp_length = batch
            probs = model(sent, atten_mask, seg_ids=seg_id, length=tmp_length).squeeze()
        else:
            sent, atten_mask, tmp_labels = batch
            _, probs = model(sent, atten_mask)
        # sent, seg_id, atten_mask, tmp_labels, tmp_length = batch
        # probs = model(sent, atten_mask, seg_ids=seg_id, length=tmp_length)

        # if hps.loss_func == "CrossEntropy":
        #     # predictions += torch.argmax(probs, 1).cpu().numpy().tolist()
        #     predictions += torch.argmax(torch.cat((probs[::2].unsqueeze(1), probs[1::2].unsqueeze(1)), 1), 1).cpu().tolist()
        #     labels += tmp_labels.cpu().tolist()
        #     loss += loss_function(probs, tmp_labels.float()).item()
        # else:
        predictions += probs.squeeze().cpu().tolist()
        loss += loss_function(probs, tmp_labels.float()).item()
        labels += tmp_labels.cpu().numpy().tolist()

    # if hps.loss_func == 'CrossEntropy':
    #     count = 0
    #     for i in range(len(predictions)):
    #         if predictions[i] == labels[i]:
    #             count += 1
    #         else:
    #             continue

    #     return count/len(labels), loss
    # else:
    if hps.data_name == 'commonsenseqa':
        a1 = torch.FloatTensor(predictions[::5]).unsqueeze(1)
        a2 = torch.FloatTensor(predictions[1::5]).unsqueeze(1)
        a3 = torch.FloatTensor(predictions[2::5]).unsqueeze(1)
        a4 = torch.FloatTensor(predictions[3::5]).unsqueeze(1)
        a5 = torch.FloatTensor(predictions[4::5]).unsqueeze(1)
        a = torch.cat((a1, a2, a3, a4, a5), dim=1)

        t_a1 = torch.FloatTensor(labels[::5]).unsqueeze(1)
        t_a2 = torch.FloatTensor(labels[1::5]).unsqueeze(1)
        t_a3 = torch.FloatTensor(labels[2::5]).unsqueeze(1)
        t_a4 = torch.FloatTensor(labels[3::5]).unsqueeze(1)
        t_a5 = torch.FloatTensor(labels[4::5]).unsqueeze(1)
        t_a = torch.cat((t_a1, t_a2, t_a3, t_a4, t_a5), dim=1)
        predict_labels = torch.argmax(a, 1).tolist()
        true_labels = torch.argmax(t_a, 1).tolist()

    elif hps.data_name == 'because':
        # softmax = nn.Softmax(1)
        a = predictions
        t_a = labels
        predict_labels = torch.sigmoid(torch.FloatTensor(a)).tolist()
        true_labels = t_a
        for k, p in enumerate(predict_labels):
            if p >= 0.5:
                predict_labels[k] = 1
            else:
                predict_labels[k] = 0

    elif hps.data_name == 'event_storyline':
        a = predictions
        predict_labels = torch.sigmoid(torch.FloatTensor(a)).tolist()
        predict_labels = [1 if p >= 0.5 else 0 for p in predict_labels]
        t_a = labels
        tp, tn, fp, fn = 0, 0, 0, 0
        for k in range(len(t_a)):
            if labels[k] == 1 and predict_labels[k] == 1:
                tp += 1
            elif labels[k] == 1 and predict_labels[k] == 0:
                fn += 1
            elif labels[k] == 0 and predict_labels[k] == 1:
                fp += 1
            else:
                tn += 1
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = 2*precision*recall/(precision+recall)
        return f1, 0

    else:
        a1 = torch.FloatTensor(predictions[::2]).unsqueeze(1)
        a2 = torch.FloatTensor(predictions[1::2]).unsqueeze(1)
        a = torch.cat((a1, a2), dim=1)
        t_a1 = torch.FloatTensor(labels[::2]).unsqueeze(1)
        t_a2 = torch.FloatTensor(labels[1::2]).unsqueeze(1)
        t_a = torch.cat((t_a1, t_a2), dim=1)
        predict_labels = torch.argmax(a, 1).tolist()
        true_labels = torch.argmax(t_a, 1).tolist()
    
    
    count = 0
    for i in range(len(predict_labels)):
        if predict_labels[i] == true_labels[i]:
            count += 1
        else:
            continue
    return count/len(true_labels), loss


def define_logger():
    logger = logging.getLogger('Discriminate logger')
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

    return logger, formatter


def tokenize_gen(data, hps):
    if hps.model_name == 'bart':
        tokenizer = BartTokenizer.from_pretrained(hps.model_dir)
    elif hps.model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer = None

    inputs = []
    labels = []
    premise = []
    for example in data:
        if hps.model_name == 'bart':
            seq1 = example['cause'] + example['effect']
            seq2 = example['general_truth']
            inputs.append(seq1)
            labels.append(seq2)
        elif hps.model_name == 'gpt2':
            inputs.append([example['cause']+' '+example['effect'], example['general_truth']])
            premise.append(example['cause']+' '+example['effect'])
            labels.append(example['general_truth'])
        else:
            return

    if hps.model_name == 'bart':
        outputs = tokenizer(inputs, padding=True)
        input_ids = torch.LongTensor(outputs['input_ids'])
        input_attention_mask = torch.LongTensor(outputs['attention_mask'])
        label_output = tokenizer(labels, padding=True)
        label_ids = torch.LongTensor(label_output['input_ids'])
        label_attention_mask = torch.LongTensor(label_output['attention_mask'])

        return input_ids, input_attention_mask, label_ids, label_attention_mask

    elif hps.model_name == 'gpt2':
        evaluate_outputs = tokenizer(labels, padding=True, return_token_type_ids=True)
        labels_ids = torch.LongTensor(evaluate_outputs['input_ids'])
        labels_mask = torch.LongTensor(evaluate_outputs['attention_mask'])
        labels_seg_id = torch.LongTensor(evaluate_outputs['token_type_ids'])

        tokenizer.padding_side = 'left'
        outputs = tokenizer(inputs, padding=True, return_token_type_ids=True)
        input_ids = torch.LongTensor(outputs['input_ids'])
        input_attention_mask = torch.LongTensor(outputs['attention_mask'])
        input_seg_id = torch.LongTensor(outputs['token_type_ids'])

        premise_outputs = tokenizer(premise, padding=True, return_token_type_ids=True)
        premise_ids = torch.LongTensor(premise_outputs['input_ids'])
        premise_mask = torch.LongTensor(premise_outputs['attention_mask'])
        premise_seg_ids = torch.LongTensor(premise_outputs['token_type_ids'])
        return input_ids, input_attention_mask, input_seg_id, labels_ids, labels_mask, labels_seg_id, premise_ids, premise_mask, premise_seg_ids


def evaluation_bart(dataloader, model, hps):
    tokenizer = BartTokenizer.from_pretrained(hps.model_dir)
    score = 0
    for batch in dataloader:
        if hps.cuda:
            batch = tuple(term.cuda() for term in batch)

        input_ids, input_mask, labels, label_mask = batch
        predict_id = torch.zeros([input_ids.shape[0], 1]).long().cuda()
        decoder_ids = torch.zeros([input_ids.shape[0], 1]).long().cuda()

        while decoder_ids.shape[1] < 35 and predict_id.tolist() not in [[[2], [2]], [[1], [1]]]:
            output = model(input_ids, input_mask=input_mask, decoder_ids=decoder_ids, mode='test')
            predict_id = torch.argmax(output[0][:, -1, :], -1).unsqueeze(1)

            decoder_ids = torch.cat((decoder_ids, predict_id), -1)

        label_tokens = [tokenizer.convert_ids_to_tokens(labels[i]) for i in range(labels.shape[0])]
        predict_tokens = [tokenizer.convert_ids_to_tokens(decoder_ids[i]) for i in range(decoder_ids.shape[0])]
        references = [tokenizer.convert_tokens_to_string(tokens) for tokens in label_tokens]
        hypothesis = [tokenizer.convert_tokens_to_string(tokens) for tokens in predict_tokens]
        references = [remove_special_tokens(text) for text in references]
        hypothesis = [remove_special_tokens(text) for text in hypothesis]

        score += sum([bleu([references[i]], hypothesis[i]) for i in range(len(references))])

    return score / len(dataloader) / hps.batch_size


def evaluate_gpt2(dataloader, model, hps):
    tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)
    score = 0
    for batch in dataloader:
        if hps.cuda:
            batch = tuple(term.cuda() for term in batch)

        gen_ids, gen_mask, _, premise_ids, premise_mask, premise_token_type_ids = batch
        decode_ids = torch.zeros([premise_ids.shape[0], 1]).long().cuda()
        predict_id = torch.zeros([premise_ids.shape[0], 1]).long().cuda()

        while decode_ids.shape[1] <= 35 and predict_id.tolist() != (torch.ones([hps.batch_size, 1]).long()*50256).tolist():
            output = model(premise_ids, premise_mask, token_type_ids=premise_token_type_ids, mode='test')
            predict_id = torch.argmax(output[1][:, -1, :], -1).unsqueeze(1)
            decode_ids = torch.cat((decode_ids, predict_id), -1)
            premise_ids = torch.cat((premise_ids, predict_id), -1)
            premise_mask = torch.cat((premise_mask, torch.ones([premise_mask.shape[0], 1]).long().cuda()), -1)
            premise_token_type_ids = torch.cat((premise_token_type_ids, torch.ones([premise_token_type_ids.shape[0], 1]).long().cuda()), -1)

        label_tokens = [tokenizer.convert_ids_to_tokens(gen_ids[i]) for i in range(gen_ids.shape[0])]
        predict_tokens = [tokenizer.convert_ids_to_tokens(decode_ids[i][1:]) for i in range(decode_ids.shape[0])]
        references = [tokenizer.convert_tokens_to_string(tokens) for tokens in label_tokens]
        hypothesis = [tokenizer.convert_tokens_to_string(tokens) for tokens in predict_tokens]
        references = [remove_special_tokens(text) for text in references]
        hypothesis = [remove_special_tokens(text) for text in hypothesis]

        score += sum([bleu([references[i]], hypothesis[i]) for i in range(len(references))])

    return score / len(dataloader) / hps.batch_size


def remove_special_tokens(text):
    return text.replace('<s>', '').replace('</s>', '').replace('<pad>', '').replace('<unk>', '').replace('<|endoftext|>', '')



def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


# def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=0.7, top_k=40, device='cuda', sample=True, attention_mask=None):
#     if start_token is None:
#         assert context is not None, 'Specify exactly one of start_token and context!'

#         context = torch.tensor(context, device=device, dtype=torch.long)
#     else:
#         assert context is None, 'Specify exactly one of start_token and context!'
#         context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
#     prev = context
#     output = context.cuda()
#     past = None
#     with torch.no_grad():
#         for i in trange(length):
#             logits, past = model(output, attention_mask, past_key_values=None, mode='test')[1:]
#             logits = logits[:, -1, :] / temperature
#             logits = top_k_logits(logits, k=top_k)
#             log_probs = F.softmax(logits, dim=-1)
#             if sample:
#                 prev = torch.multinomial(log_probs, num_samples=1)
#             else:
#                 _, prev = torch.topk(log_probs, k=1, dim=-1)
#             output = torch.cat((output, prev), dim=1)
#             attention_mask = torch.cat((attention_mask, torch.ones(prev.shape).long().cuda()), -1)
#     return output


def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=0.7, top_k=40,
                    device='cuda', sample=True, attention_mask=None, input_type='ids'):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        if input_type == 'ids':
            context = torch.tensor(context, device=device, dtype=torch.long)
        else:
            context = torch.tensor(context, device=device)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output_id = None
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            if input_type == 'ids':
                gen_output = model(input_ids=output, attention_mask=attention_mask, past_key_values=None, mode='test')
                logits = gen_output['logits']
            else:
                logits, past, hiddens = model(inputs_embeds=output, attention_mask=attention_mask, past_key_values=None, output_hidden_states=True)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            if input_type == 'ids':
                output = torch.cat((output, prev), dim=1)
            else:
                output = torch.cat((output, hiddens[-1][:, -1, :].unsqueeze(1)), 1)
                output_id = prev if output_id is None else torch.cat((output_id, prev), 1)

            attention_mask = torch.cat((attention_mask, torch.ones(prev.shape).long().cuda()), -1)
    return output if input_type == 'ids' else output_id


def gpt2_evaluate(model, length, data_loader, hps):
    tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)

    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    rouge1p, rouge1r, rouge1f, rouge2p, rouge2r, rouge2f, rougelp, rougelr, rougelf = 0, 0, 0, 0, 0, 0, 0, 0, 0
    rouge = Rouge()
    output_text = []
    nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    for batch in data_loader:
        if hps.cuda:
            batch = tuple(term.cuda() for term in batch)
        gen_ids, gen_mask, _, premise_ids, premise_mask, premise_token_type_ids = batch

        # output = sample_sequence(model, length, device='cuda', context=premise_ids, batch_size=hps.batch_size, attention_mask=premise_mask, input_type='ids')
        generated = model.generate(input_ids=premise_ids, 
                                   attention_mask=premise_mask, 
                                   max_length=length+premise_ids.shape[1], 
                                   num_beams=5, 
                                   early_stopping=True, 
                                   do_sample=True,
                                   no_repeat_ngram_size=3,
                                   repetition_penalty=1.5
                                   )

        # generated = output[:, premise_ids.shape[1]:]
        # pdb.set_trace()
        generated = generated[:, premise_ids.shape[1]:]

        generated_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated.cpu().tolist()]
        gold_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in gen_ids.cpu().tolist()]
        input_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in premise_ids]
        output_text += [[input_text[i], gold_text[i], generated_text[i].split('.')[0]+'.'] for i in range(len(input_text))]


        for i in range(generated.shape[0]):
            # predict_tokens = tokenizer.convert_ids_to_tokens(generated[i])
            # generated_text = remove_special_tokens(tokenizer.convert_tokens_to_string(predict_tokens))


            # gold_tokens = tokenizer.convert_ids_to_tokens(gen_ids[i])
            # gold_text = remove_special_tokens(tokenizer.convert_tokens_to_string(gold_tokens))

            bleu1 += bleu([gold_text[i]], generated_text[i].split('.')[0]+'.', [1, 0, 0, 0])
            bleu2 += bleu([gold_text[i]], generated_text[i].split('.')[0]+'.', [0, 1, 0, 0])
            bleu3 += bleu([gold_text[i]], generated_text[i].split('.')[0]+'.', [0, 0, 1, 0])
            bleu4 += bleu([gold_text[i]], generated_text[i].split('.')[0]+'.', [0, 0, 0, 1])

            try:
                scores = rouge.get_scores(generated_text[i], gold_text[i])
                rouge1 = scores[0]['rouge-1']
                rouge1f += rouge1['f']
                rougelp += rouge1['p']
                rouge1r += rouge1['r']

                rouge2 = scores[0]['rouge-2']
                rouge2f += rouge2['f']
                rouge1p += rouge2['p']
                rouge2r += rouge2['r']

                rougel = scores[0]['rouge-l']
                rougelf += rougel['f']
                rougelp += rougel['p']
                rougelr += rougel['r']
            except:
                continue

    num_instances = (len(data_loader)-1) * hps.batch_size + gen_ids.shape[0]

    fo = open(hps.output_dir+'/gpt2_predict_'+nowtime+'.csv', 'w', encoding='utf-8')
    writer = csv.writer(fo)
    writer.writerows(output_text)

    return bleu1/num_instances, bleu2/num_instances, bleu3/num_instances, bleu4/num_instances, rouge1r/num_instances, rouge2r/num_instances, rougelr/num_instances



def bart_evaluate(model, data_loader, hps):
    tokenizer = BartTokenizer.from_pretrained(hps.model_dir)

    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    rouge1p, rouge1r, rouge1f, rouge2p, rouge2r, rouge2f, rougelp, rougelr, rougelf = 0, 0, 0, 0, 0, 0, 0, 0, 0
    # rouge1p, rouge1r, rouge1f, rouge2p, rouge2r, rouge2f, rougelp, rougelr, rougelf = 0, 0, 0, 0, 0, 0, 0, 0, 0
    rouge = Rouge()
    nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_text = []

    for batch in data_loader:
        if hps.cuda:
            batch = tuple(term.cuda() for term in batch)

        input_ids, input_mask, labels, label_mask = batch
        generate_ids = model.generate(input_ids, 
        							  attention_mask=input_mask, 
        							  num_beams=hps.beam_size, 
        							  max_length=hps.length, 
        							  early_stopping=True, 
                                      no_repeat_ngram_size=3,
        							  repetition_penalty=1.5,        							  
        							  # temperature=0.7,
        							  # length_penalty=0.6
                                      )
        # generate_ids = generate_ids[:, input_ids.shape[1]:]

        generate_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generate_ids]
        gold_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in labels]
        input_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in input_ids]

        output_text += [[input_text[i], gold_text[i], generate_text[i].split('.')[0]+'.'] for i in range(len(input_text))]


        for i in range(len(gold_text)):

            bleu1 += bleu([gold_text[i]], generate_text[i].split('.')[0]+'.', [1, 0, 0, 0])
            bleu2 += bleu([gold_text[i]], generate_text[i].split('.')[0]+'.', [0, 1, 0, 0])
            bleu3 += bleu([gold_text[i]], generate_text[i].split('.')[0]+'.', [0, 0, 1, 0])
            bleu4 += bleu([gold_text[i]], generate_text[i].split('.')[0]+'.', [0, 0, 0, 1])

            try:
                scores = rouge.get_scores(generate_text[i], gold_text[i])
            except:
                scores = [
                  {
                    "rouge-1": {
                      "f": 0.0,
                      "p": 0.0,
                      "r": 0.0
                    },
                    "rouge-2": {
                      "f": 0.0,
                      "p": 0.0,
                      "r": 0.0
                    },
                    "rouge-l": {
                      "f": 0.0,
                      "p": 0.0,
                      "r": 0.0
                    }
                  }
                ]
            rouge1 = scores[0]['rouge-1']
            rouge1f += rouge1['f']
            rougelp += rouge1['p']
            rouge1r += rouge1['r']

            rouge2 = scores[0]['rouge-2']
            rouge2f += rouge2['f']
            rouge1p += rouge2['p']
            rouge2r += rouge2['r']

            rougel = scores[0]['rouge-l']
            rougelf += rougel['f']
            rougelp += rougel['p']
            rougelr += rougel['r']

    num_instances = (len(data_loader)-1) * hps.batch_size + input_ids.shape[0]

    fo = open(hps.output_dir+'/bart_predict_'+nowtime+'.csv', 'w', encoding='utf-8')
    writer = csv.writer(fo)
    writer.writerows(output_text)

    return bleu1/num_instances, bleu2/num_instances, bleu3/num_instances, bleu4/num_instances, rouge1r/num_instances, rouge2r/num_instances, rougelr/num_instances
























