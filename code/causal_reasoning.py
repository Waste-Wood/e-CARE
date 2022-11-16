import jsonlines
import json
import sys


def read_gold_data(path):
    fi = jsonlines.open(path, 'r')
    labels = {line['index']: line['label'] for line in fi}
    return labels


def evaluation_metrics(gold, predictions):
    count = 0
    for key, value in predictions.items():
        try:
            gold_label = gold[key]
        except:
            raise KeyError('{} is not a correct index in the dataset.')

        if gold_label == value:
            count += 1
        else:
            continue

    accuracy = count / len(gold)
    return accuracy


def main():
    prediction_file = sys.argv[1]
    gold_file = sys.argv[2]

    predictions = json.load(open(prediction_file, 'r'))
    gold_labels = read_gold_data(gold_file)

    fo = open('./prediction_causal_reasoning.json', 'w')

    accuracy = evaluation_metrics(gold_labels, predictions)
    json.dump({"accuracy": accuracy}, fo)
    print("[Accuracy]: {}".format(accuracy))


if __name__ == '__main__':
    main()




