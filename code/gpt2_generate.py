import argparse
from utils.utils import load_data, define_logger, tokenize_gen, evaluate_gpt2, gpt2_evaluate, compute_ppl
import random
import numpy as np
import torch
from model.generatively_model import gpt2_generate, bart_generate
from transformers import AdamW, GPT2LMHeadModel
import sys
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
import datetime
import logging
import pdb


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
    parser.add_argument('--train', type=str, default='train_gen.pkl', help='The train data directory')
    parser.add_argument('--dev', type=str, default='dev_gen.pkl', help='The dev data directory')
    parser.add_argument('--test', type=str, default='test_gen.pkl', help='The test data directory')

    # Model Settings
    parser.add_argument('--model_name', type=str, default='gpt2', help='Pretrained model name')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use gpu for training')
    parser.add_argument('--gpu', type=str, default='0', help='Gpu ids for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size for training and evaluation')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle training data')
    parser.add_argument('--epochs', type=int, default=200, help='training iterations')
    parser.add_argument('--evaluation_step', type=int, default=100,
                        help='when training for some steps, start evaluation')
    parser.add_argument('--lr', type=float, default=1e-5, help='the learning rate of training')
    parser.add_argument('--set_seed', type=bool, default=True, help='Whether to fix the random seed')
    parser.add_argument('--seed', type=int, default=1024, help='fix the random seed for reproducible')
    parser.add_argument('--patient', type=int, default=10, help='the patient of early-stopping')
    parser.add_argument('--length', type=int, default=22, help='the max length of generated text')
    parser.add_argument('--output_dir', type=str, default='./output/output_examples')

    # parsing the hyper-parameters from command line and define logger
    hps = parser.parse_args()
    logger, formatter = define_logger()
    nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(hps.log_dir, 'generated_'+hps.model_name+'.txt')
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
    train_data = load_data(os.path.join(hps.data_dir, hps.train))
    dev_data = load_data(os.path.join(hps.data_dir, hps.dev))
    test_data = load_data(os.path.join(hps.data_dir, hps.test))

    # Tokenization
    logger.info("[INFO] Tokenization and Padding for Data")
    train_ids, train_mask, train_seg_ids, train_label_ids, train_label_mask, _, _, _, _ = tokenize_gen(train_data, hps)
    _, _, _, dev_label_ids, dev_label_mask, dev_label_seg_ids, dev_premise_ids, dev_premise_mask, dev_premise_seg_ids = tokenize_gen(dev_data, hps)
    _, _, _, test_label_ids, test_label_mask, test_label_seg_ids, test_premise_ids, test_premise_mask, test_premise_seg_ids = tokenize_gen(test_data, hps)

    # Dataset and DataLoader
    logger.info("[INFO] Creating Dataset and splitting batch for data")
    TRAIN = TensorDataset(train_ids, train_mask, train_seg_ids, train_label_ids, train_label_mask)
    DEV = TensorDataset(dev_label_ids, dev_label_mask, dev_label_seg_ids, dev_premise_ids, dev_premise_mask, dev_premise_seg_ids)
    TEST = TensorDataset(test_label_ids, test_label_mask, test_label_seg_ids, test_premise_ids, test_premise_mask, test_premise_seg_ids)
    train_dataloader = DataLoader(TRAIN, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    dev_dataloader = DataLoader(DEV, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    test_dataloader = DataLoader(TEST, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)

    # initialize model, optimizer, loss_function
    logger.info('[INFO] Loading pretrained model, setting optimizer and loss function')

    # model = gpt2_generate(hps)
    model = GPT2LMHeadModel.from_pretrained(hps.model_dir)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)

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

            input_ids, input_mask, input_seg_ids, input_labels, input_labels_mask = batch
            # pdb.set_trace()
            tmp = torch.ones(input_labels_mask.shape).long()
            count_mask_length = torch.sum(tmp==input_labels_mask.cpu(), 1).squeeze().tolist()
            true_labels = None
            for j in range(input_ids.shape[0]):
                if true_labels is None:
                    # true_labels = torch.cat((torch.ones(count_mask_length[j]).long(), input_ids[j, count_mask_length[j]:].cpu())).unsqueeze(0)
                    true_labels = torch.cat((input_ids[j, :-count_mask_length[j]]*0-100, input_ids[j, -count_mask_length[j]:])).unsqueeze(0)
                else:
                    # true_labels = torch.cat((true_labels, torch.cat((torch.ones(count_mask_length[j]).long(), input_ids[j, count_mask_length[j]:].cpu())).unsqueeze(0)), 0)
                    true_labels = torch.cat((true_labels, torch.cat((input_ids[j, :-count_mask_length[j]]*0-100, input_ids[j, -count_mask_length[j]:])).unsqueeze(0)),0)

            # true_labels = true_labels.cuda()


            # loss = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg_ids, true_labels=true_labels, mode='train')[0]
            # pdb.set_trace()
            output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg_ids, labels=true_labels)
            loss = output[0]

            total_loss += loss.item()
            t.set_postfix(avg_loss='{}'.format(total_loss/(epoch_step+1)))
            epoch_step += 1

            loss.backward()
            optimizer.step()

            if step % hps.evaluation_step == 0 and step != 0:
                model.eval()

                with torch.no_grad():
                    print('\n')
                    logger.info("[Dev Evaluation] Start Evaluation on Dev Set")
                    dev_bleu1, dev_bleu2, dev_bleu3, dev_bleu4, dev_rouge1, dev_rouge2, dev_rougel = gpt2_evaluate(model, hps.length, dev_dataloader, hps)
                    print('\n')
                    logger.info("[Dev Metrics] Dev BLEU: \t({}, {}, {}, {})".format(dev_bleu1, dev_bleu2, dev_bleu3, dev_bleu4))
                    logger.info("[Dev Metrics] Dev Rouge: \t({}, {}, {})".format(dev_rouge1, dev_rouge2, dev_rougel))

                    dev_ppl = compute_ppl(hps, model, dev_data)
                    logger.info('[PPL] Model PerPlexity On Dev Set is {}'.format(dev_ppl))

                    if dev_bleu1 + dev_rouge1 >= best_accuracy:
                        patient = 0
                        best_accuracy = dev_bleu1 + dev_rouge1
                        logger.info("[Saving] Saving Model to {}".format(hps.save_dir))
                        # torch.save(model, os.path.join(hps.save_dir, '{}_{}'.format('generated', hps.model_name)))
                        logger.info("[Test Evaluation] Start Evaluation on Test Set")

                        test_bleu1, test_bleu2, test_bleu3, test_bleu4, test_rouge1, test_rouge2, test_rougel = gpt2_evaluate(model, hps.length, test_dataloader, hps)
                        
                        print('\n')
                        logger.info("[TEST Metrics] Test BLEU: \t({}, {}, {}, {})".format(test_bleu1, test_bleu2, test_bleu3, test_bleu4))
                        logger.info("[TEST Metrics] Test Rouge: \t({}, {}, {})".format(test_rouge1, test_rouge2, test_rougel))
                        test_ppl = compute_ppl(hps, model, test_data)
                        logger.info('[PPL] Model PerPlexity On Test Set is {}'.format(test_ppl))
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





