import torch
import torch.nn as nn
from model.multi_task_model import tokenize_data, discriminate_generate, generate_discriminate
import argparse
from utils.utils import load_data, tokenize_multi_task, evaluate_multi_task
import os
from transformers import AdamW, BartForConditionalGeneration
from tqdm import trange
from utils.utils import define_logger
import datetime
import random
from torch.utils.data import TensorDataset, DataLoader
import logging
import numpy as np
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='xCAR')

# Data Paths
parser.add_argument('--data_dir', type=str, default='./data/', help='The dataset directory')
parser.add_argument('--model_dir', type=str, default='../../huggingface_transformers/bart-base/',
                    help='The pretrained model directory')
# parser.add_argument('--generate_model_dir', type=str, default='./data/pretrained_models/gpt2')
parser.add_argument('--save_dir', type=str, default='./output/saved_model', help='The model saving directory')
parser.add_argument('--log_dir', type=str, default='./output/log', help='The training log directory')
parser.add_argument('--apex_dir', type=str, default='./output/log', help='The apex directory')

# Data names
parser.add_argument('--train', type=str, default='train.pkl', help='The train data directory')
parser.add_argument('--dev', type=str, default='dev.pkl', help='The dev data directory')
parser.add_argument('--test', type=str, default='test.pkl', help='The test data directory')

# Model Settings
parser.add_argument('--model_name', type=str, default='bart', help='Pretrained model name')
parser.add_argument('--data_name', type=str, default='copa')
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
parser.add_argument('--patient', type=int, default=20, help='the patient of early-stopping')
parser.add_argument('--hidden_size', type=int, default=100, help='the patient of early-stopping')
parser.add_argument('--n_layer', type=int, default=1, help='the patient of early-stopping')
parser.add_argument('--dropout', type=int, default=0.1, help='the patient of early-stopping')
parser.add_argument('--output_size', type=int, default=10000, help='the patient of early-stopping')
parser.add_argument('--method', type=str, default='dot', help='the patient of early-stopping')
parser.add_argument('--alpha', type=float, default=0.5, help='the weight of discriminative loss')
parser.add_argument('--length', type=int, default=22, help='the max length of generated text')
parser.add_argument('--beam_size', type=int, default=30, help='the size of beam search')
parser.add_argument('--type', type=str, default='generate_discriminate', help='the type of multi-task')

hps = parser.parse_args()
logger, formatter = define_logger()
nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = os.path.join(hps.log_dir, hps.model_name+nowtime+'.txt')
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info('[MODEL] RoBERTa-GPT2')

if hps.set_seed:
    random.seed(hps.seed)
    np.random.seed(hps.seed)
    torch.manual_seed(hps.seed)
    torch.cuda.manual_seed(hps.seed)

logger.info("[INFO] Loading Data")
train_data = load_data(os.path.join(hps.data_dir, hps.train))
dev_data = load_data(os.path.join(hps.data_dir, hps.dev))
test_data = load_data(os.path.join(hps.data_dir, hps.test))

logger.info("[INFO] Tokenization and Padding for Data")
train_ids1, train_mask1, train_ids2, train_mask2, train_labels = tokenize_data(hps, train_data)
dev_ids1, dev_mask1, dev_ids2, dev_mask2, dev_labels = tokenize_data(hps, dev_data)
test_ids1, test_mask1, test_ids2, test_mask2, test_labels = tokenize_data(hps, test_data)

logger.info("[INFO] Creating Dataset and splitting batch for data")
TRAIN_input = TensorDataset(train_ids1, train_mask1, train_labels)
TRAIN_output = TensorDataset(train_ids2, train_mask2)
DEV_input = TensorDataset(dev_ids1, dev_mask1, dev_labels)
DEV_output = TensorDataset(dev_ids2, dev_mask2)
TEST_input = TensorDataset(test_ids1, test_mask1, test_labels)
TEST_output = TensorDataset(test_ids2, test_mask2)
train_dataloader_input = DataLoader(TRAIN_input, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
train_dataloader_output = DataLoader(TRAIN_output, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
dev_dataloader_input = DataLoader(DEV_input, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
dev_dataloader_output = DataLoader(DEV_output, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
test_dataloader_input = DataLoader(TEST_input, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
test_dataloader_output = DataLoader(TEST_output, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)

logger.info('[INFO] Loading pretrained model, setting optimizer and loss function')
# model = BartForConditionalGeneration.from_pretrained(hps.model_dir)
model = discriminate_generate(hps) if hps.type == 'discriminate_generate' else generate_discriminate(hps)
if hps.cuda:
    model = model.cuda()
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
loss_function = nn.BCEWithLogitsLoss(reduction='mean')
loss_function2 = nn.CrossEntropyLoss(reduction='mean')

logger.info("[INFO] Start Training")
step = 0
patient = 0
best_accuracy = 0
stop_train = False

for epoch in range(hps.epochs):
    logger.info('[Epoch] {}'.format(epoch))
    t = trange(len(train_dataloader_input))
    epoch_step = 0
    total_loss = 0
    for i, batch1, batch2 in zip(t, train_dataloader_input, train_dataloader_output):
        optimizer.zero_grad()
        model.train()
        if hps.cuda:
            batch1 = tuple(term.cuda() for term in batch1)
            batch2 = tuple(term.cuda() for term in batch2)

        input_ids, attention_mask, labels = batch1
        decoder_ids, decoder_mask = batch2
        scores, gen_logits = model(input_ids,
                                   attention_mask,
                                   decoder_ids,
                                   decoder_mask,
                                   decoder_ids,
                                   mode='train')

        discriminate_loss = loss_function(scores, labels)
        index = torch.argmax(torch.cat((scores[::2].unsqueeze(1), scores[1::2].unsqueeze(1)), 1), 1)
        gen_logits = torch.cat((gen_logits[::2].unsqueeze(1), gen_logits[1::2].unsqueeze(1)), 1)
        gen_logits = gen_logits[range(gen_logits.shape[0]), index, :, :]
        decoder_ids = torch.cat((decoder_ids[::2].unsqueeze(1), decoder_ids[1::2].unsqueeze(1)), 1)
        decoder_ids = decoder_ids[range(decoder_ids.shape[0]), index, :]
        generate_loss = loss_function2(gen_logits.view(-1, gen_logits.shape[-1]), decoder_ids.view(-1))
            # discriminate_loss = loss_function(scores, labels)
            # generate_loss = loss_function2(gen_logits.view(-1, gen_logits.shape[-1]), decoder_ids.view(-1))
        loss = hps.alpha * discriminate_loss + (1-hps.alpha) * generate_loss

        total_loss += loss.item()
        t.set_postfix(avg_loss='{}'.format(total_loss/(epoch_step+1)))
        epoch_step += 1
        loss.backward()
        optimizer.step()

        if step % hps.evaluation_step == 0 and step != 0:
            model.eval()

            with torch.no_grad():
                logger.info("[Dev Evaluation] Strain Evaluation on Dev Set")

                dev_accu, dev_bleu1, dev_bleu2, dev_bleu3, dev_bleu4 = evaluate_multi_task(model,
                                                                                           dev_dataloader_input,
                                                                                           dev_dataloader_output,
                                                                                           hps)
                logger.info('[DEV METRICS]')
                logger.info('[ACCURACY]:\t{}'.format(dev_accu))
                logger.info('[BLEU]:\t{}-{}-{}-{}'.format(dev_bleu1, dev_bleu2, dev_bleu3, dev_bleu4))
                if dev_accu + dev_bleu1 > best_accuracy:
                    best_accuracy = dev_accu + dev_bleu1
                    patient = 0
                    logger.info("[Saving] Saving Model to {}".format(hps.save_dir))
                    torch.save(model, os.path.join(hps.save_dir, '{}_{}'.format(hps.model_name, best_accuracy)))
                    test_accu, test_bleu1, test_bleu2, test_bleu3, test_bleu4 = evaluate_multi_task(model,
                                                                                                    dev_dataloader_input,
                                                                                                    dev_dataloader_output,
                                                                                                    hps)
                    logger.info('[TEST METRICS]')
                    logger.info('[ACCURACY]:\t{}'.format(test_accu))
                    logger.info('[BLEU]:\t{}-{}-{}-{}'.format(test_bleu1, test_bleu2, test_bleu3, test_bleu4))
                else:
                    patient += 1

                if patient >= hps.patient:
                    stop_train = True
                    break

        step += 1

    if stop_train:
        break

