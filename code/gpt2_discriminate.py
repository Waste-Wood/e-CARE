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
import copy
import torch.nn.functional as F



class gpt2_discriminate(nn.Module):
    def __init__(self, hps):
        super(gpt2_discriminate, self).__init__()
        self.hps = hps
        self.model = GPT2LMHeadModel.from_pretrained(hps.model_dir)
        self.linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, pos):
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # pdb.set_trace()
        hidden_state = outputs.hidden_states[-1]
        pos = pos.squeeze().unsqueeze(0)
        hidden_state = hidden_state[range(hidden_state.shape[0]), pos, :].squeeze(0)
        logits = self.linear(hidden_state).squeeze(-1)
        return logits



def tokenization(data, hps):
    tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)

    inputs = []
    labels = []
    pos = []

    for example in data:
        if not hps.hyp_only:
            if example['ask-for'] == 'cause':
                inputs.append([example['alternative1'], example['premise']])
                inputs.append([example['alternative2'], example['premise']])
            else:
                inputs.append([example['premise'], example['alternative1']])
                inputs.append([example['premise'], example['alternative2']])
        else:
            inputs += [example['alternative1'], example['alternative2']]
        labels += [0, 1] if example['label'] == 1 else [1, 0]
    outputs = tokenizer(inputs, return_length=True)
    input_ids = outputs['input_ids']
    attention_mask = outputs['attention_mask']
    length = outputs['length']
    max_length = max(length)
    for i in range(len(input_ids)):
        gap = max_length - len(input_ids[i]) + 1
        pos.append(len(input_ids[i]))
        input_ids[i] += [50256 for _ in range(gap)]
        attention_mask[i] += [1] + [0 for _ in range(gap-1)]
    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(pos), torch.LongTensor(labels)



def evaluate(hps, model, dataloader, loss_function, optimizer):
    predictions, attack_predictions = [], []
    labels = []
    loss = 0
    attack_loss = 0
    
    # model.eval()
    for batch in dataloader:
        attack_model = copy.deepcopy(model)
        # attack_model.eval()
        optimizer.zero_grad()
        if hps.cuda:
            batch = tuple(term.cuda() for term in batch)

        input_ids, attention_mask, pos, tmp_labels = batch
        logits = model(input_ids, attention_mask=attention_mask, pos=pos)

        # predictions += logits.cpu().tolist()
        tmp_loss = loss_function(logits, tmp_labels.float())
        loss += tmp_loss.item()
        # labels += tmp_labels.cpu().numpy().tolist()

        tmp_loss.backward()
        embedding_grad = optimizer.param_groups[0]['params'][0].grad
        with torch.no_grad():
            model.eval()
            logits = model(input_ids, attention_mask=attention_mask, pos=pos)
            predictions += logits.cpu().tolist()
            tmp_loss = loss_function(logits, tmp_labels.float())
            loss += tmp_loss.item()
            labels += tmp_labels.cpu().numpy().tolist()

            state_dict = attack_model.state_dict()
            # embedding_grad = F.softmax(embedding_grad, 1)
            attack_embedding = torch.sum(embedding_grad * embedding_grad, -1)
            attack_embedding = attack_embedding.pow(0.5).unsqueeze(-1)
            attack_embedding = attack_embedding.pow(-1) * embedding_grad
            state_dict['model.transformer.wte.weight'] += hps.attack_rate * attack_embedding
            attack_model.load_state_dict(state_dict)
            # pdb.set_trace()
            attack_model.eval()
            attack_logits = attack_model(input_ids, attention_mask=attention_mask, pos=pos)
            attack_predictions += attack_logits.cpu().tolist()
            attack_loss += loss_function(attack_logits, tmp_labels.float()).item()


    a1 = torch.FloatTensor(predictions[::2]).unsqueeze(1)
    a2 = torch.FloatTensor(predictions[1::2]).unsqueeze(1)
    a = torch.cat((a1, a2), dim=1)
    predict_labels = torch.argmax(a, 1).tolist()

    t_a1 = torch.FloatTensor(labels[::2]).unsqueeze(1)
    t_a2 = torch.FloatTensor(labels[1::2]).unsqueeze(1)
    t_a = torch.cat((t_a1, t_a2), dim=1)
    true_labels = torch.argmax(t_a, 1).tolist()

    a_t1 = torch.FloatTensor(attack_predictions[::2]).unsqueeze(1)
    a_t2 = torch.FloatTensor(attack_predictions[1::2]).unsqueeze(1)
    a_t = torch.cat((a_t1, a_t2), 1)
    attack_predict_labels = torch.argmax(a_t, 1)

    count, attack_count = 0, 0
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
    return count/len(true_labels), loss, attack_count/len(true_labels), attack_loss



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
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size for training and evaluation')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle training data')
    parser.add_argument('--epochs', type=int, default=200, help='training iterations')
    parser.add_argument('--evaluation_step', type=int, default=100,
                        help='when training for some steps, start evaluation')
    parser.add_argument('--lr', type=float, default=1e-5, help='the learning rate of training')
    parser.add_argument('--set_seed', type=bool, default=True, help='Whether to fix the random seed')
    parser.add_argument('--seed', type=int, default=1024, help='fix the random seed for reproducible')
    parser.add_argument('--patient', type=int, default=10, help='the patient of early-stopping')
    parser.add_argument('--length', type=int, default=20, help='the max length of generated text')
    parser.add_argument('--output_dir', type=str, default='./output/output_examples')
    parser.add_argument('--hyp_only', type=bool, default=False)
    parser.add_argument('--attack_rate', type=float, default=0.015)

    # parsing the hyper-parameters from command line and define logger
    hps = parser.parse_args()
    logger, formatter = define_logger()
    # nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if hps.hyp_only:
        log_path = os.path.join(hps.log_dir, 'discriminate_' + hps.model_name + '_hyp_only.txt')
    else:
        log_path = os.path.join(hps.log_dir, 'discriminate_' + hps.model_name + '.txt')
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
    logger.info("[INFO] Hypothesis Only:\t{}".format(hps.hyp_only))
    train_data = load_data(os.path.join(hps.data_dir, hps.train))
    dev_data = load_data(os.path.join(hps.data_dir, hps.dev))
    test_data = load_data(os.path.join(hps.data_dir, hps.test))

    # Tokenization
    logger.info("[INFO] Tokenization and Padding for Data")
    train_ids, train_mask, train_pos, train_labels = tokenization(train_data, hps)
    dev_ids, dev_mask, dev_pos, dev_labels = tokenization(dev_data, hps)
    test_ids, test_mask, test_pos, test_labels = tokenization(test_data, hps)

    # Dataset and DataLoader
    logger.info("[INFO] Creating Dataset and splitting batch for data")
    TRAIN = TensorDataset(train_ids, train_mask, train_pos, train_labels)
    DEV = TensorDataset(dev_ids, dev_mask, dev_pos, dev_labels)
    TEST = TensorDataset(test_ids, test_mask, test_pos, test_labels)
    train_dataloader = DataLoader(TRAIN, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    dev_dataloader = DataLoader(DEV, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    test_dataloader = DataLoader(TEST, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)

    # initialize model, optimizer, loss_function
    logger.info('[INFO] Loading pretrained model, setting optimizer and loss function')

    model = gpt2_discriminate(hps)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')

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

            input_ids, input_mask, pos, label = batch

            logits = model(input_ids, attention_mask=input_mask, pos = pos)
            # pdb.set_trace()
            loss = loss_function(logits, label.float())
            
            total_loss += loss.item()
            t.set_postfix(avg_loss='{}'.format(total_loss / (epoch_step + 1)))
            epoch_step += 1

            loss.backward()
            optimizer.step()

            if step % hps.evaluation_step == 0 and step != 0:
                model.eval()

                # with torch.no_grad():
                    # print('\n')
                logger.info("[Dev Evaluation] Start Evaluation on Dev Set")
                evaluation_output  = evaluate(hps, model, dev_dataloader, loss_function, optimizer)
                # print('\n')
                logger.info("[Dev Metrics] Dev Accuracy: \t{}".format(evaluation_output[0]))
                logger.info("[Dev Metrics] Dev Attack Accuracy: \t{}".format(evaluation_output[2]))
                logger.info("[Dev Metrics] Dev Loss: \t{}".format(evaluation_output[1]))
                logger.info("[Dev Metrics] Dev Attack Loss: \t{}".format(evaluation_output[3]))


                if evaluation_output[0] >= best_accuracy:
                    patient = 0
                    best_accuracy = evaluation_output[0]
                    logger.info("[Saving] Saving Model to {}".format(hps.save_dir))
                    # torch.save(model, os.path.join(hps.save_dir, '{}_{}'.format('generated', hps.model_name)))
                    logger.info("[Test Evaluation] Start Evaluation on Test Set")

                    evaluation_output = evaluate(hps, model, test_dataloader, loss_function, optimizer)

                    print('\n')
                    logger.info("[Test Metrics] Test Accuracy: \t{}".format(evaluation_output[0]))
                    logger.info("[Test Metrics] Test Attack Accuracy: \t{}".format(evaluation_output[2]))
                    logger.info("[Test Metrics] Test Loss: \t{}".format(evaluation_output[1]))
                    logger.info("[Test Metrics] Test Attack Loss: \t{}".format(evaluation_output[3]))
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




















