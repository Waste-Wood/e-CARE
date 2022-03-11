from transformers import RobertaConfig, RobertaModel, AdamW, RobertaTokenizer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils.utils import load_data, quick_tokenize, define_logger, evaluation
import os
import random
import argparse
import datetime
import logging
import numpy as np
from tqdm import trange
import math
from random import choice
import pickle


class roberta(nn.Module):
    def __init__(self, hps):
        super(roberta, self).__init__()
        self.model = RobertaModel.from_pretrained(hps.model_dir)
        self.config = RobertaConfig.from_pretrained(hps.model_dir)
        self.hps = hps
        self.linear = nn.Linear(self.config.hidden_size, 1)
        self.softmax = nn.Softmax(1)

    def forward(self, input_ids, attention_mask, mode='train'):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output[1]
        scores = self.linear(cls_token).squeeze(1)
        if mode == 'train':
            alternative1 = scores[::2].unsqueeze(1)
            alternative2 = scores[1::2].unsqueeze(1)
            alternatives = torch.cat((alternative1, alternative2), 1)
            probs = self.softmax(alternatives)
        else:
            probs = None
        return probs, scores


def data_process(path):
    train_data = load_data(os.path.join(path, 'train.pkl'))
    test_data = load_data(os.path.join(path, 'test.pkl'))
    dev_data = load_data(os.path.join(path, 'dev.pkl'))

    data = train_data + test_data + dev_data
    return data


def split_data(data):
    t, d = [], []
    for i in range(len(data)):
        if i % 10 < 7:
            t.append(data[i])
        else:
            d.append(data[i])
    return t, d



def shuffle_data(data, seed):
    random.seed(seed)
    random.shuffle(data)

    cause_label_0, cause_label_1, effect_label_0, effect_label_1 = [], [], [], []

    train, dev = [], []

    for d in data:
        if d['label'] == 0 and d['ask-for'] == 'cause':
            cause_label_0.append(d)
        elif d['label'] == 1 and d['ask-for'] == 'cause':
            cause_label_1.append(d)
        elif d['label'] == 0:
            effect_label_0.append(d)
        else:
            effect_label_1.append(d)

    t, d = split_data(cause_label_0)
    train += t
    dev += d

    t, d = split_data(cause_label_1)
    train += t
    dev += d

    t, d = split_data(effect_label_0)
    train += t
    dev += d

    t, d = split_data(effect_label_1)
    train += t
    dev += d

    random.shuffle(train)
    random.shuffle(dev)

    if len(train) % 2 != 0 and len(dev) % 2 != 0:
        train.append(dev[-1])
        dev = dev[:-1]
    elif len(train) % 2 != 0:
        train = train[:-1]
    elif len(dev) % 2 != 0:
        dev = dev[:-1]

    right_causes, right_effects, wrong_causes, wrong_effects, causes, effects, ask_fors = [], [], [], [], [], [], []
    truths, labels = [], []
    for example in dev:
        ask_for = example['ask-for']
        if ask_for == 'cause':
            effects.append(example['premise'])
            right_causes.append(example['alternative1'] if example['label'] == 0 else example['alternative2'])
            wrong_causes.append(example['alternative2'] if example['label'] == 0 else example['alternative1'])
        else:
            causes.append(example['premise'])
            right_effects.append(example['alternative1'] if example['label'] == 0 else example['alternative2'])
            wrong_effects.append(example['alternative2'] if example['label'] == 0 else example['alternative1'])
        ask_fors.append(ask_for)
        truths.append(example['general_truth'])
        labels.append(example['label'])

    return train, dev, right_causes, right_effects, wrong_causes, wrong_effects, causes, effects, ask_fors, truths, labels


def training(train, dev, model, loss_function, optimizer, hps):
    step = 0
    patient = 0
    best_accuracy = 0
    stop_train = False
    final_model = None
    for epoch in range(hps.epochs):
        logger.info('[Epoch] {}'.format(epoch))
        t = trange(len(train))
        epoch_step = 0
        total_loss = 0
        for i, batch in zip(t, train):
            optimizer.zero_grad()
            model.train()
            if hps.cuda:
                batch = tuple(term.cuda(0) for term in batch)

            sent, atten_mask, labels = batch
            probs, scores = model(sent, atten_mask)

            # labels1 = labels[::2].unsqueeze(1)
            # labels2 = labels[1::2].unsqueeze(1)
            # labels = torch.cat((labels1, labels2), 1)
            # labels = torch.argmax(labels, 1)
            loss = loss_function(scores, labels.float())

            total_loss += loss.item()
            t.set_postfix(avg_loss='{}'.format(total_loss / (epoch_step + 1)))
            epoch_step += 1

            loss.backward()
            optimizer.step()

            if step % hps.evaluation_step == 0 and step != 0:
                model.eval()

                with torch.no_grad():
                    print('\n')
                    logger.info("[Dev Evaluation] Start Evaluation on Dev Set")

                    dev_accu, dev_loss = evaluation(hps, dev, model, loss_function, mode='filtering')
                    print('\n')
                    logger.info("[Dev Metrics] Dev Accuracy: \t{}".format(dev_accu))
                    logger.info("[Dev Metrics] Dev Loss: \t{}".format(dev_loss))

                    if dev_accu >= best_accuracy:
                        patient = 0
                        best_accuracy = dev_accu
                        logger.info("[Saving] Saving Model to {}".format(hps.save_dir))
                        final_model = model
                        # torch.save(model, os.path.join(hps.save_dir, '{}_{}'.format(hps.model_name, dev_accu)))
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
    return final_model


def adversarial_filtering(module, ask_fors, causes, effects, right_causes, right_effects, wrong_causes, wrong_effects,
                          t, tokenizer, truths, labels, iterations, hps):
    module.eval()
    candidates = []
    count = 0
    for i, ask_for, tq in zip(range(len(ask_fors)), ask_fors, trange(len(ask_fors))):
        if ask_for == 'cause':
            h_positive = [right_causes[ask_fors[:i].count(ask_for)], effects[ask_fors[:i].count(ask_for)]]
            h_negative = [wrong_causes[ask_fors[:i].count(ask_for)], effects[ask_fors[:i].count(ask_for)]]
            h_negatives = [[cause, effects[ask_fors[:i].count(ask_for)]] for cause in wrong_causes]
            h_negatives += [[cause, effects[ask_fors[:i].count(ask_for)]] for cause in (right_causes[:ask_fors[:i].count(ask_for)]+right_causes[ask_fors[:i].count(ask_for)+2:])]
            while h_positive in h_negatives:
                h_negatives.remove(h_positive)
        else:
            h_positive = [causes[ask_fors[:i].count(ask_for)], right_effects[ask_fors[:i].count(ask_for)]]
            h_negative = [causes[ask_fors[:i].count(ask_for)], wrong_effects[ask_fors[:i].count(ask_for)]]
            h_negatives = [[causes[ask_fors[:i].count(ask_for)], effect] for effect in wrong_effects]
            h_negatives += [[causes[ask_fors[:i].count(ask_for)], effect] for effect in (right_effects[:ask_fors[:i].count(ask_for)]+right_effects[ask_fors[:i].count(ask_for)+2:])]
            # h_negatives.remove(h_positive)
            while h_positive in h_negatives:
                h_negatives.remove(h_positive)
        random.shuffle(h_negatives)
        h_negatives = h_negatives[: len(h_negatives)//4]
        input1 = tokenizer([h_positive]+[h_negative]+h_negatives, padding=True)
        input_ids, attention_mask = torch.LongTensor(input1['input_ids']), torch.LongTensor(input1['attention_mask'])
        dataset = TensorDataset(input_ids, attention_mask)
        data_loader = DataLoader(dataset, batch_size=hps.batch_size, drop_last=False, shuffle=False)
        # print(len(data_loader))
        scores = None
        for batch in data_loader:
            if hps.cuda:
                batch = tuple(term.cuda(0) for term in batch)

            ids, mask = batch
            # print('IDS shape: {}'.format(ids.shape))
            _, score = module(ids, mask, mode='filtering')
            scores = score if scores is None else torch.cat((scores, score), 0)

        positive = scores[0]
        negative = scores[1]
        negatives = scores[2:]
        delta = (positive - negative).cpu().item()
        candidate_deltas = -1 * (negatives - positive.cpu().item())
        candidate_deltas = candidate_deltas.cpu().tolist()
        if delta >= 0:
            indexes = list(filter(lambda x: candidate_deltas[x] < 0, range(len(candidate_deltas))))
        else:
            indexes = list(filter(lambda x: candidate_deltas[x] < delta, range(len(candidate_deltas))))

        if len(indexes) == 0:
            indexes = list(filter(lambda x: candidate_deltas[x] < delta, range(len(candidate_deltas))))
        if len(indexes) == 0 or iterations >= 12:
            indexes = list(filter(lambda x: candidate_deltas[x] < delta/2, range(len(candidate_deltas))))
        r = random.random()
        if t < r or delta < 0 or len(indexes) <= 0:
            candidates.append([h_negative[0]] + h_positive if ask_for == 'cause' else h_positive + [h_negative[1]])
        else:
            index = choice(indexes)
            count += 1
            candidates.append([h_negatives[index][0]] + h_positive if ask_for == 'cause' else h_positive + [h_negatives[index][1]])
    dev_adversarial_filtered = []
    for k, example in enumerate(candidates):
        if ask_fors[k] == 'cause':
            if labels[k] == 0:
                dev_adversarial_filtered.append({'index': k,
                                                 'general_truth': truths[k],
                                                 'ask-for': ask_fors[k],
                                                 'premise': candidates[k][2],
                                                 'alternative1': candidates[k][1],
                                                 'alternative2': candidates[k][0],
                                                 'label': labels[k]
                                                 })
            else:
                dev_adversarial_filtered.append({'index': k,
                                                 'general_truth': truths[k],
                                                 'ask-for': ask_fors[k],
                                                 'premise': candidates[k][2],
                                                 'alternative1': candidates[k][0],
                                                 'alternative2': candidates[k][1],
                                                 'label': labels[k]
                                                 })
        else:
            if labels[k] == 0:
                dev_adversarial_filtered.append({'index': k,
                                                 'general_truth': truths[k],
                                                 'ask-for': ask_fors[k],
                                                 'premise': candidates[k][0],
                                                 'alternative1': candidates[k][1],
                                                 'alternative2': candidates[k][2],
                                                 'label': labels[k]
                                                 })
            else:
                dev_adversarial_filtered.append({'index': k,
                                                 'general_truth': truths[k],
                                                 'ask-for': ask_fors[k],
                                                 'premise': candidates[k][0],
                                                 'alternative1': candidates[k][2],
                                                 'alternative2': candidates[k][1],
                                                 'label': labels[k]
                                                 })
    return dev_adversarial_filtered, count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xCAR')

    # Data Paths
    parser.add_argument('--data_dir', type=str, default='./data/final_data/raw/', help='The dataset directory')
    parser.add_argument('--model_dir', type=str, default='../../huggingface_transformers/roberta-base/',
                        help='The pretrained model directory')
    parser.add_argument('--save_dir', type=str, default='./output/saved_model', help='The model saving directory')
    parser.add_argument('--log_dir', type=str, default='./output/log', help='The training log directory')
    parser.add_argument('--apex_dir', type=str, default='./output/log', help='The apex directory')

    # Data names
    parser.add_argument('--train', type=str, default='train.pkl', help='The train data directory')
    parser.add_argument('--dev', type=str, default='dev.pkl', help='The dev data directory')
    parser.add_argument('--test', type=str, default='test.pkl', help='The test data directory')

    # Model Settings
    parser.add_argument('--model_name', type=str, default='roberta', help='Pretrained model name')
    parser.add_argument('--data_name', type=str, default='copa')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use gpu for training')
    parser.add_argument('--gpu', type=str, default='0 1', help='Gpu ids for training')
    # parser.add_argument('--apex', type=bool, default=False, help='Whether to use half precision')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size for training and evaluation')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle training data')
    parser.add_argument('--epochs', type=int, default=100, help='training iterations')
    parser.add_argument('--evaluation_step', type=int, default=20,
                        help='when training for some steps, start evaluation')
    parser.add_argument('--lr', type=float, default=1e-5, help='the learning rate of training')
    parser.add_argument('--set_seed', type=bool, default=True, help='Whether to fix the random seed')
    parser.add_argument('--seed', type=int, default=1004, help='fix the random seed for reproducible')
    parser.add_argument('--patient', type=int, default=3, help='the patient of early-stopping')
    parser.add_argument('--loss_func', type=str, default='BCEWithLogitsLoss', help="loss function of output")
    parser.add_argument('--data_path', type=str, default='./data/final_data/raw', help="loss function of output")
    parser.add_argument('--iterations', type=int, default=50, help="loss function of output")
    parser.add_argument('--t_e', type=float, default=0.2, help="loss function of output")
    parser.add_argument('--t_s', type=float, default=1.0, help="loss function of output")
    parser.add_argument('--hyp_only', type=bool, default=False, help="loss function of output")
    parser.add_argument('--append', type=bool, default=True, help="append examples before adversarial_filtering")

    hps = parser.parse_args()
    logger, formatter = define_logger()
    nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(hps.log_dir, 'adversarial_filtering_' + hps.model_name + nowtime + '.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if hps.set_seed:
        random.seed(hps.seed)
        np.random.seed(hps.seed)
        torch.manual_seed(hps.seed)
        torch.cuda.manual_seed(hps.seed)

    data = data_process(hps.data_path)
    tokenizer = RobertaTokenizer.from_pretrained(hps.model_dir)

    logger.info('[APPEND] \t{}'.format(hps.append))

    for i in range(hps.iterations):
        
        logger.info('[TRAINING START]')
        train, dev, right_causes, right_effects, wrong_causes, wrong_effects, causes, effects, ask_fors, truths, labels = shuffle_data(
            data, hps.seed)

        train_ids, train_mask, _, train_labels, _ = quick_tokenize(train, hps)
        dev_ids, dev_mask, _, dev_labels, _ = quick_tokenize(dev, hps)

        TRAIN = TensorDataset(train_ids, train_mask, train_labels)
        DEV = TensorDataset(dev_ids, dev_mask, dev_labels)

        train_dataloader = DataLoader(TRAIN, batch_size=hps.batch_size, drop_last=False)
        dev_dataloader = DataLoader(DEV, batch_size=hps.batch_size, drop_last=False)

        model = roberta(hps)
        # loss_function = nn.CrossEntropyLoss(reduction='mean')
        loss_function = nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
        if hps.cuda:
            model = model.cuda(0)
            model = nn.DataParallel(model, device_ids=[0, 1])

        t_i = hps.t_e + (hps.t_s - hps.t_e) / (1+math.exp(0.3*i-0.9*hps.iterations/4))
        module = training(train_dataloader, dev_dataloader, model, loss_function, optimizer, hps)

        logger.info('[FILTERING START]')
        logger.info('[FILTERING] {}'.format(i))
        logger.info('[Temperature]: {}'.format(t_i))
        with torch.no_grad():
            dev_adversarial_filtered, count = adversarial_filtering(
                module,
                ask_fors,
                causes,
                effects,
                right_causes,
                right_effects,
                wrong_causes,
                wrong_effects,
                t_i,
                tokenizer,
                truths,
                labels,
                i,
                hps
            )
        logger.info('[INFO] {}/{} instances has been replaced'.format(count, len(ask_fors)))
        logger.info('='*50)

        if hps.append:
            random.shuffle(dev)
            data = train + dev_adversarial_filtered + dev[:200]
            fo = open('./data/final_data/adversarial_filtering_append/data_{}_{}.pkl'.format(i, t_i), 'wb')
        else:
            data = train + dev_adversarial_filtered
            fo = open('./data/final_data/adversarial_filtering/data_{}_{}.pkl'.format(i, t_i), 'wb')
        pickle.dump(data, fo)


















