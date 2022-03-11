import argparse
from utils.utils import load_data, define_logger, tokenize_gen, evaluate_gpt2, gpt2_evaluate, compute_ppl
import random
import numpy as np
import torch
from model.generatively_model import gpt2_generate, bart_generate
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer
import sys
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
import datetime
import logging
import pdb
from rouge import Rouge
from nltk import bleu
import csv
import copy
import torch.nn.functional as F


class gpt2_multi_task(nn.Module):
    def __init__(self, hps):
        super(gpt2_multi_task, self).__init__()
        self.hps = hps
        self.model = GPT2LMHeadModel.from_pretrained(hps.model_dir)
        self.linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, pos, mode='train', token_type_ids=None):
        # if mode == 'train':

        if mode == 'train':
            # pdb.set_trace()
            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True, token_type_ids=token_type_ids)
            hidden_state = outputs.hidden_states[-1]
            # pos = pos.squeeze().unsqueeze(0)
            hidden_state = hidden_state[range(hidden_state.shape[0]), pos, :]
            logits = self.linear(hidden_state).squeeze(-1)
            gen_logits = outputs.logits
            return logits, gen_logits

        if mode == 'test':
            output_dict = self.model.generate(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              max_length=self.hps.length + input_ids.shape[1],
                                              num_beams=self.hps.beam_size,
                                              early_stopping=self.hps.early_stopping,
                                              # do_sample=self.hps.do_sample,
                                              no_repeat_ngram_size=self.hps.no_repeat_ngram_size,
                                              repetition_penalty=self.hps.repetition_penalty,
                                              output_hidden_states=True,
                                              return_dict_in_generate=True,
                                              output_scores=True
                                              )

            if self.hps.mode == 'generate_discriminate':
                # pdb.set_trace()
                scores = output_dict.scores
                hiddens = output_dict.hidden_states

                hiddens = [h[-1][:, -1, :].view(input_ids.shape[0], self.hps.beam_size, -1) for h in hiddens]
                scores = [s.view(input_ids.shape[0], self.hps.beam_size, -1) for s in scores]
                index = [s.argmax(-1) for s in scores]

                generated_text = output_dict.sequences[:, input_ids.shape[1]:]
                index = [index[i].eq(generated_text[:, i].unsqueeze(1)) for i in range(len(index))]
                selected = None
                # pdb.set_trace()
                for i in range(len(index)):
                    ttmp = None
                    for j in range(index[i].shape[0]):
                        tmp = hiddens[i][j][index[i][j]]
                        if tmp.shape[0] == 0:
                            tmp = hiddens[i][j][0]
                        else:
                            tmp = tmp[0]
                        ttmp = tmp.unsqueeze(0) if ttmp is None else torch.cat((ttmp, tmp.unsqueeze(0)), 0)
                    selected = ttmp.unsqueeze(1) if selected is None else torch.cat((selected, ttmp.unsqueeze(1)), 1)

                # pdb.set_trace()
                # eos_mask = generated_text.eq(self.model.config.eos_token_id)
                eos_mask = generated_text.eq(13)

                gen_ids = generated_text
                for i in range(eos_mask.shape[0]):
                    flag = False
                    if eos_mask[i].sum().item() == 1:
                        continue
                    elif eos_mask[i].sum().item() == 0:
                        eos_mask[i][-1] = True
                        continue
                    else:
                        for j, t in enumerate(eos_mask[i]):
                            if t.item():
                                if flag:
                                    eos_mask[i][j] = False
                                flag = True
                            else:
                                continue

                # pdb.set_trace()
                # selected = selected.view(eos_mask.shape[0], eos_mask.shape[1], -1)[eos_mask, :]
                selected = selected[eos_mask, :]
                logits = self.linear(selected).squeeze()
                gen_ids = generated_text

            else:
                # pdb.set_trace()
                outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_state = outputs.hidden_states[-1]
                # pos = pos.squeeze().unsqueeze(0)
                hidden_state = hidden_state[range(hidden_state.shape[0]), pos, :]
                logits = self.linear(hidden_state).squeeze(-1)
                # logits = torch.cat((logits[::2].unsqueeze(1), logits[1::2].unsqueeze(1)), 1)
                gen_ids = output_dict.sequences[:, input_ids.shape[1]:]

            return logits, gen_ids


def tokenization(data, hps):
    tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    inputs = []
    labels = []
    pos1 = []
    pos2 = []
    loss_label = []
    truth = []
    # token_type_ids = []

    for example in data:
        if example['ask-for'] == 'cause':
            inputs.append([example['alternative1'], example['premise']])
            inputs.append([example['alternative2'], example['premise']])
        else:
            inputs.append([example['premise'], example['alternative1']])
            inputs.append([example['premise'], example['alternative2']])
        truth += [example['general_truth']] * 2
        labels += [0, 1] if example['label'] == 1 else [1, 0]

    outputs = tokenizer(inputs, return_length=True)
    input_ids = outputs['input_ids']
    attention_mask = outputs['attention_mask']
    length = outputs['length']
    # max_length = max(length)

    truth_output = tokenizer(truth, return_length=True)
    truth_ids = truth_output['input_ids']
    truth_mask = truth_output['attention_mask']
    truth_length = truth_output['length']

    premise_ids = []
    premise_mask = []

    for i in range(len(input_ids)):
        pos1.append(len(input_ids[i]))
        input_ids[i] += [50256]
        attention_mask[i] += [1]
        loss_label.append([-100 for _ in range(len(input_ids[i]))])
        # token_type_ids += [0 for _ in range(len(input_ids[i]))]

        gap1 = max(length) + 1 - len(input_ids[i])
        premise_ids.append(input_ids[i] + [50256 for _ in range(gap1)])
        premise_mask.append(attention_mask[i] + [0 for _ in range(gap1)])

        input_ids[i] += truth_ids[i]
        attention_mask[i] += truth_mask[i]
        # token_type_ids += [1 for _ in range(len(truth_ids[i]))]
        pos2.append(len(input_ids[i])-1)

        loss_label[i] += truth_ids[i]

        gap2 = max(truth_length) - len(truth_ids[i])
        truth_ids[i] += [50256 for _ in range(gap2)]

    truth_ids = torch.LongTensor(truth_ids)

    max_length = max([length[k]+truth_length[k] for k in range(len(length))]) + 2
    for i in range(len(input_ids)):
        gap = max_length - len(input_ids[i])
        input_ids[i] += [50256 for _ in range(gap)]
        attention_mask[i] += [0 for _ in range(gap)]
        # token_type_ids += [0 for _ in range(gap)]
        loss_label[i] += [-100 for _ in range(gap)]

    # premise_outputs = tokenizer(inputs, padding=True)
    premise_ids = torch.LongTensor(premise_ids)
    premise_mask = torch.LongTensor(premise_mask)

    if hps.mode == 'discriminate_generate':
        pos = pos1
    else:
        pos = pos2
    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(pos), torch.LongTensor(
        labels), torch.LongTensor(loss_label), premise_ids, premise_mask, truth_ids


def evaluate(hps, model, dataloader, loss_function, loss_function2, optimizer):
    
    tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)
    # predictions = []
    labels = []
    predict_labels, attack_predict_labels = [], []
    loss = 0
    attack_loss = 0

    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    rouge1p, rouge1r, rouge1f, rouge2p, rouge2r, rouge2f, rougelp, rougelr, rougelf = 0, 0, 0, 0, 0, 0, 0, 0, 0
    rouge = Rouge()
    output_text = []
    
    # model.eval()
    # pdb.set_trace()
    for batch in dataloader:
        attack_model = copy.deepcopy(model)
        
        # attack_model.eval()
        optimizer.zero_grad()
        if hps.cuda:
            batch = tuple(term.cuda() for term in batch)

        ids, mask, pos, tmp_labels, loss_label, input_ids, attention_mask, truth = batch
        dis_logits, gen_logits = model(ids, attention_mask=mask, pos=pos, mode='train')
        
        # predictions += logits.cpu().tolist()
        tmp_loss = loss_function(dis_logits, tmp_labels.float())
        loss += tmp_loss.cpu().item()


        t_1 = tmp_labels[::2].unsqueeze(1)
        t_2 = tmp_labels[1::2].unsqueeze(1)
        t_ = torch.cat((t_1, t_2), 1)
        t_index = t_.argmax(1)

        g1 = gen_logits[::2].unsqueeze(1)
        g2 = gen_logits[1::2].unsqueeze(1)
        g = torch.cat((g1, g2), 1)

        l1 = loss_label[::2].unsqueeze(1)
        l2 = loss_label[1::2].unsqueeze(1)
        l = torch.cat((l1, l2), 1)

        g = g[range(g.shape[0]), t_index, :]
        l = l[range(l.shape[0]), t_index, :]

        # pdb.set_trace()

        shift_logits = g[..., :-1, :].contiguous()
        shift_labels = l[..., 1:].contiguous()

        loss_gen = loss_function2(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_loss = hps.alpha * tmp_loss + (1 - hps.alpha) * loss_gen
        total_loss.backward()
        embedding_grad = optimizer.param_groups[0]['params'][0].grad
        # attack
        

        # optimizer.zero_grad()
        with torch.no_grad():
            # pdb.set_trace()
            model.eval()
            state_dict = attack_model.state_dict()
            logits, gen_ids = model(input_ids, attention_mask=attention_mask, pos=pos, mode='test')
            # embedding_grad = F.softmax(embedding_grad, 1)
            attack_embedding = torch.sum(embedding_grad * embedding_grad, -1)
            attack_embedding = attack_embedding.pow(0.5).unsqueeze(-1)
            attack_embedding = attack_embedding.pow(-1) * embedding_grad
            state_dict['model.transformer.wte.weight'] += hps.attack_rate * attack_embedding
            attack_model.load_state_dict(state_dict)
            attack_model.eval()
            attack_logits, _ = attack_model(input_ids, attention_mask=attention_mask, pos=pos, mode='test')

            attack_loss += loss_function(attack_logits, tmp_labels.float()).item()
            labels += tmp_labels.cpu().numpy().tolist()

        # generated = gen_ids[:, input_ids.shape[1]:]
            tmp_predict = logits.cpu().tolist()
            a1 = torch.FloatTensor(tmp_predict[::2]).unsqueeze(1)
            a2 = torch.FloatTensor(tmp_predict[1::2]).unsqueeze(1)
            a = torch.cat((a1, a2), dim=1)
            predict_label = torch.argmax(a, 1)
            predict_labels += predict_label.cpu().tolist()

            attack_tmp_predict = attack_logits.cpu().tolist()
            a_t1 = torch.FloatTensor(attack_tmp_predict[::2]).unsqueeze(1)
            a_t2 = torch.FloatTensor(attack_tmp_predict[1::2]).unsqueeze(1)
            a_t = torch.cat((a_t1, a_t2), 1)
            attack_predict_label = torch.argmax(a_t, 1)
            attack_predict_labels += attack_predict_label.cpu().tolist()

            g1 = gen_ids[::2].unsqueeze(1)
            g2 = gen_ids[1::2].unsqueeze(1)
            g = torch.cat((g1, g2), 1)

            t_1 = tmp_labels[::2].unsqueeze(1)
            t_2 = tmp_labels[1::2].unsqueeze(1)
            t = torch.cat((t_1, t_2), 1)

        # generated = g[range(g.shape[0]), predict_label]
            generated = g[range(g.shape[0]), t.argmax(1)]

            generated_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                              generated.cpu().tolist()]

            gold_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                         truth[::2].cpu().tolist()]
            # input_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in input_ids]
            output_text += [[gold_text[i], generated_text[i].split('.')[0] + '.'] for i in range(len(gold_text))]
            for i in range(generated.shape[0]):
                bleu1 += bleu([gold_text[i]], generated_text[i].split('.')[0] + '.', [1, 0, 0, 0])
                bleu2 += bleu([gold_text[i]], generated_text[i].split('.')[0] + '.', [0, 1, 0, 0])
                bleu3 += bleu([gold_text[i]], generated_text[i].split('.')[0] + '.', [0, 0, 1, 0])
                bleu4 += bleu([gold_text[i]], generated_text[i].split('.')[0] + '.', [0, 0, 0, 1])

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

    # predict_labels = predict_labels.cpu().tolist()
    # pdb.set_trace()
    t_a1 = torch.FloatTensor(labels[::2]).unsqueeze(1)
    t_a2 = torch.FloatTensor(labels[1::2]).unsqueeze(1)
    t_a = torch.cat((t_a1, t_a2), dim=1)
    true_labels = torch.argmax(t_a, 1).tolist()
    count = 0
    attack_count = 0
    for i in range(len(predict_labels)):
        if predict_labels[i] == true_labels[i] and attack_predict_labels[i] == true_labels[i]:
            count += 1
            attack_count += 1
        elif predict_labels[i] == true_labels[i]:
            count += 1
        elif attack_predict_labels[i] == true_labels[i]:
            attack_count += 1
        else:
            continue
    # pdb.set_trace()
    nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fo = open(hps.output_dir + '/gpt2_predict_' + nowtime + '.csv', 'w', encoding='utf-8')
    writer = csv.writer(fo)
    writer.writerows(output_text)

    # num_instances = int(((len(data_loader)-1) * hps.batch_size + truth_ids.shape[0]) / 2)

    return count / len(true_labels), bleu1 / len(true_labels), bleu2 / len(true_labels), bleu3 / len(
        true_labels), bleu4 / len(true_labels), rouge1r / len(true_labels), rouge2r / len(true_labels), rougelr / len(
        true_labels), loss, attack_count / len(true_labels), attack_loss


def compute_ppl(hps, model, data):
    # pdb.set_trace()
    tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)
    lls = []
    total_length = 0
    for example in data:
        if example['ask-for'] == 'cause':
            input_text = (example['alternative1'] + ' ' + example['premise']) if example['label'] == 0 else (example['alternative2'] + ' ' + example['premise'])
        else:
            input_text = (example['premise'] + ' ' + example['alternative1']) if example['label'] == 0 else (example['premise'] + ' ' + example['alternative2'])
        truth = example['general_truth']
        inputs = tokenizer(input_text)
        input_ids = torch.LongTensor(inputs['input_ids']+[50256]).unsqueeze(0).cuda()
        attention_mask = torch.LongTensor(inputs['attention_mask']+[1]).unsqueeze(0).cuda()
        label_inputs = tokenizer(truth)
        label_ids = torch.LongTensor(label_inputs['input_ids']).unsqueeze(0).cuda()
        length = label_ids.shape[1]
        total_length += length
        
        # label_mask = torch.LongTensor(label_inputs['attention_mask']).unsqueeze(0).cuda()
        attention_mask = torch.cat((attention_mask, torch.ones(1, label_ids.shape[1]).long().cuda()), 1)
        label_ids = torch.cat((torch.LongTensor([-100]*input_ids.shape[1]).unsqueeze(0).cuda(), label_ids), 1)
        input_ids = torch.cat((input_ids, label_ids[:, input_ids.shape[1]:]), 1)
        with torch.no_grad():
            loss = model.model(input_ids, attention_mask=attention_mask, labels=label_ids)[0]
            lls.append(loss * length)
    # pdb.set_trace()
    ppl = torch.exp(torch.stack(lls).sum() / total_length)

    return ppl.item()


def main():
    parser = argparse.ArgumentParser(description='xCAR')

    # Data Paths
    parser.add_argument('--data_dir', type=str, default='./data/', help='The dataset directory')
    parser.add_argument('--model_dir', type=str, default='../../huggingface_transformers/gpt2/',
                        help='The pretrained model directory')
    parser.add_argument('--save_dir', type=str, default='./output/saved_model', help='The model saving directory')
    parser.add_argument('--log_dir', type=str, default='./output/log', help='The training log directory')
    parser.add_argument('--apex_dir', type=str, default='./output/log', help='The apex directory')

    # Data names
    parser.add_argument('--train', type=str, default='train.pkl', help='The train data directory')
    parser.add_argument('--dev', type=str, default='dev.pkl', help='The dev data directory')
    parser.add_argument('--test', type=str, default='test.pkl', help='The test data directory')

    # Model Settings
    parser.add_argument('--model_name', type=str, default='gpt2', help='Pretrained model name')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use gpu for training')
    parser.add_argument('--gpu', type=str, default='0', help='Gpu ids for training')
    # parser.add_argument('--apex', type=bool, default=False, help='Whether to use half precision')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training and evaluation')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle training data')
    parser.add_argument('--epochs', type=int, default=200, help='training iterations')
    parser.add_argument('--evaluation_step', type=int, default=2,
                        help='when training for some steps, start evaluation')
    parser.add_argument('--lr', type=float, default=1e-5, help='the learning rate of training')
    parser.add_argument('--set_seed', type=bool, default=True, help='Whether to fix the random seed')
    parser.add_argument('--seed', type=int, default=1024, help='fix the random seed for reproducible')
    parser.add_argument('--patient', type=int, default=10, help='the patient of early-stopping')
    parser.add_argument('--length', type=int, default=20, help='the max length of generated text')
    parser.add_argument('--output_dir', type=str, default='./output/output_examples')
    parser.add_argument('--hyp_only', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--mode', type=str, default='discriminate_generate')
    # parser.add_argument('--length', type=int, default=22)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('--repetition_penalty', type=float, default=1.5)
    parser.add_argument('--early_stopping', type=bool, default=False)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--attack_rate', type=float, default=0.015)

    # parsing the hyper-parameters from command line and define logger
    hps = parser.parse_args()
    logger, formatter = define_logger()
    # nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(hps.log_dir, hps.mode + '_' + hps.model_name + '.txt')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # fix random seed
    if hps.set_seed:
        random.seed(hps.seed)
        np.random.seed(hps.seed)
        torch.manual_seed(hps.seed)
        torch.cuda.manual_seed(hps.seed)

    # load data
    # logger.info("[Pytorch] %s", torch.)
    logger.info("[INFO] Loading Data")
    logger.info("[INFO] Mode:\t{}".format(hps.mode))
    train_data = load_data(os.path.join(hps.data_dir, hps.train))
    dev_data = load_data(os.path.join(hps.data_dir, hps.dev))
    test_data = load_data(os.path.join(hps.data_dir, hps.test))

    # Tokenization
    logger.info("[INFO] Tokenization and Padding for Data")
    train_ids, train_mask, train_pos, train_labels, train_loss_labels, _, _, _ = tokenization(train_data, hps)
    dev, dev_mask, dev_pos, dev_labels, dev_loss_labels, dev_premise_ids, dev_premise_mask, dev_truth = tokenization(dev_data, hps)
    test, test_mask, test_pos, test_labels, test_loss_labels, test_premise_ids, test_premise_mask, test_truth = tokenization(test_data, hps)

    # Dataset and DataLoader
    logger.info("[INFO] Creating Dataset and splitting batch for data")
    TRAIN = TensorDataset(train_ids, train_mask, train_pos, train_labels, train_loss_labels)
    DEV = TensorDataset(dev, dev_mask, dev_pos, dev_labels, dev_loss_labels, dev_premise_ids, dev_premise_mask, dev_truth)
    TEST = TensorDataset(test, test_mask, test_pos, test_labels, test_loss_labels, test_premise_ids, test_premise_mask, test_truth)
    train_dataloader = DataLoader(TRAIN, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    dev_dataloader = DataLoader(DEV, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    test_dataloader = DataLoader(TEST, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)

    # initialize model, optimizer, loss_function
    logger.info('[INFO] Loading pretrained model, setting optimizer and loss function')

    model = gpt2_multi_task(hps)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    loss_function2 = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)

    # Multi-Gpu training
    if hps.cuda:
        gpu_ids = [int(x) for x in hps.gpu.split(' ')]
        model.cuda(gpu_ids[0])
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)

    # training
    logger.info("[INFO] Start Training")
    step = 0
    patient = 0
    best_accuracy = 0
    stop_train = False

    for epoch in range(hps.epochs):
        logger.info('[Epoch] {}'.format(epoch))
        t = trange(len(train_dataloader))
        epoch_step = 0
        total_loss = 0
        for i, batch in zip(t, train_dataloader):
            optimizer.zero_grad()
            model.train()
            if hps.cuda:
                batch = tuple(term.cuda() for term in batch)

            input_ids, input_mask, pos, label, loss_label = batch

            dis_logits, gen_logits = model(input_ids, attention_mask=input_mask, pos=pos)
            # pdb.set_trace()
            loss_dis = loss_function(dis_logits, label.float())

            # a1 = dis_logits[::2].unsqueeze(1)
            # a2 = dis_logits[1::2].unsqueeze(1)
            # a = torch.cat((a1, a2), 1)
            # index = a.argmax(1)

            t_1 = label[::2].unsqueeze(1)
            t_2 = label[1::2].unsqueeze(1)
            t_ = torch.cat((t_1, t_2), 1)
            t_index = t_.argmax(1)

            g1 = gen_logits[::2].unsqueeze(1)
            g2 = gen_logits[1::2].unsqueeze(1)
            g = torch.cat((g1, g2), 1)

            l1 = loss_label[::2].unsqueeze(1)
            l2 = loss_label[1::2].unsqueeze(1)
            l = torch.cat((l1, l2), 1)

            g = g[range(g.shape[0]), t_index, :]
            l = l[range(l.shape[0]), t_index, :]

            # pdb.set_trace()

            shift_logits = g[..., :-1, :].contiguous()
            shift_labels = l[..., 1:].contiguous()

            loss_gen = loss_function2(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = hps.alpha * loss_dis + (1 - hps.alpha) * loss_gen

            total_loss += loss.item()
            t.set_postfix(avg_loss='{}'.format(total_loss / (epoch_step + 1)))
            epoch_step += 1

            loss.backward()
            optimizer.step()

            if step % hps.evaluation_step == 0 and step != 0:
                model.eval()

                # with torch.no_grad():
                print('\n')
                logger.info("[Dev Evaluation] Start Evaluation on Dev Set")
                evaluate_output = evaluate(hps, model, dev_dataloader, loss_function, loss_function2, optimizer)
                dev_ppl = compute_ppl(hps, model, dev_data)
                logger.info("[Dev Metrics] Dev Perplexity: \t{}".format(dev_ppl))
                print('\n')
                logger.info("[Dev Metrics] Dev Accuracy: \t{}".format(evaluate_output[0]))
                logger.info("[DEV Metrics] Dev Attack Accuracy: \t{}".format(evaluate_output[-2]))
                logger.info(
                    "[Dev Metrics] Dev BLEU:\t({}, {}, {}, {})".format(evaluate_output[1], evaluate_output[2],
                                                                       evaluate_output[3], evaluate_output[4]))
                logger.info("[Dev Metrics] Dev Discriminate Loss: \t{}".format(evaluate_output[-3]))
                logger.info("[Dev Metrics] Dev Discriminate Attack Loss: \t{}".format(evaluate_output[-1]))
                logger.info(
                    "[Dev Metrics] Dev Rouge Recall:\t({}, {}, {})".format(evaluate_output[5], evaluate_output[6],
                                                                           evaluate_output[7]))

                if evaluate_output[0] >= best_accuracy:
                    patient = 0
                    best_accuracy = evaluate_output[0]
                    logger.info("[Saving] Saving Model to {}".format(hps.save_dir))
                    # torch.save(model, os.path.join(hps.save_dir, '{}_{}'.format('generated', hps.model_name)))
                    logger.info("[Test Evaluation] Start Evaluation on Test Set")

                    test_output = evaluate(hps, model, test_dataloader, loss_function, loss_function2, optimizer)
                    test_ppl = compute_ppl(hps, model, test_data)
                    logger.info("[Test Metrics] Test Perplexity: \t{}".format(test_ppl))

                    print('\n')
                    logger.info("[Test Metrics] Test Accuracy: \t{}".format(test_output[0]))
                    logger.info("[Test Metrics] Test Attack Accuracy: \t{}".format(test_output[-2]))
                    logger.info("[Test Metrics] Test BLEU:\t({}, {}, {}, {})".format(test_output[1], test_output[2],
                                                                                     test_output[3],
                                                                                     test_output[4]))
                    logger.info("[Test Metrics] Test Discriminate Loss: \t{}".format(test_output[-3]))
                    logger.info("[Test Metrics] Test Discriminate Attack Loss: \t{}".format(test_output[-1]))
                    logger.info(
                        "[Test Metrics] Test Rouge Recall:\t({}, {}, {})".format(test_output[5], test_output[6],
                                                                                 test_output[7]))
                else:
                    patient += 1

                logger.info("[Patient] {}".format(patient))

                if patient >= hps.patient:
                    logger.info("[INFO] Stopping Training by Early Stopping")
                    stop_train = True
                    break
            step += 1

        if stop_train:
            break


if __name__ == '__main__':
    main()
