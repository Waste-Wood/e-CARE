from nltk import bleu
from rouge import Rouge
import jsonlines
import json
import sys


def read_gold_data(path):
    fi = jsonlines.open(path, 'r')
    explanations = {line['index']: line['conceptual_explanation'] for line in fi}
    return explanations


def evaluation_bleu(gold, predictions):
    bleu_socre = 0
    for key in predictions:
        prediction = predictions[key]
        try:
            gold = gold[key]
        except:
            raise KeyError('{} is not a correct index in e-CARE dataset'.format(key))

        bleu_socre += bleu([gold], prediction)

    avg_bleu = bleu_socre / len(gold)
    return avg_bleu


def evaluation_rouge(golds, predictions):
    rougel = 0
    rouge = Rouge()

    for key in predictions:
        prediction = predictions[key]
        try:
            gold = golds[key]
        except:
            raise KeyError('{} is not a correct index in e-CARE dataset'.format(key))

        try:
            scores = rouge.get_scores(prediction, gold)
            rougel += scores[0]['rouge-l']['r']
        except:
            continue

    avg_rougel = rougel / len(golds)
    return avg_rougel


def main():
    prediction_file = sys.argv[1]
    gold_file = sys.argv[2]

    predictions = json.load(open(prediction_file, 'r'))
    gold_labels = read_gold_data(gold_file)

    bleu_score = evaluation_bleu(gold_labels, predictions)
    rouge_l = evaluation_rouge(gold_labels, predictions)

    fo = open('./evaluation_metrics_conceptual_explanation_generation.json', 'w')

    json.dump({"bleu": bleu_score, "rouge-l": rouge_l}, fo)


if __name__ == '__main__':
    main()





