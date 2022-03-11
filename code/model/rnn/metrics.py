from nltk import bleu
from rouge import Rouge
import tqdm


gold = '/users5/kxiong/work/xCAR/data/final_data/data/tgt-test.txt'
hyp = './save/test_prediction.txt'

golden = open(gold, 'r', encoding='utf-8').readlines()

hypothesis = open(hyp, 'r', encoding='utf-8').readlines()

bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
rouge1, rouge2, rougel = 0, 0, 0
scorer = Rouge()

bar = tqdm.trange(len(golden))

for _, g, h in zip(bar, golden, hypothesis):
	bleu1 += bleu([g.strip()], h.strip(), [1, 0, 0, 0])
	bleu2 += bleu([g], h, [0, 1, 0, 0])
	bleu3 += bleu([g], h, [0, 0, 1, 0])
	bleu4 += bleu([g], h, [0, 0, 0, 1])

	scores = scorer.get_scores(h, g)

	rouge1 += scores[0]['rouge-1']['r']
	rouge2 += scores[0]['rouge-2']['r']
	rougel += scores[0]['rouge-l']['r']

print(bleu1/len(golden), bleu2/len(golden), bleu3/len(golden), bleu4/len(golden))
print(rouge1/len(golden), rouge2/len(golden), rougel/len(golden))









