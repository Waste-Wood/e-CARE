# e-CARE: a New Dataset for Exploring Explainable Causal Reasoning

## Brief Introduction
Understanding causality has vital importance for various Natural Language Processing (NLP) applications. Beyond the labeled instances, conceptual explanations of the causality can provide deep understanding of the causal fact to facilitate the causal reasoning process. We present a human-annotated explainable CAusal REasoning dataset (e-CARE), which contains over 20K causal reasoning questions, together with natural language formed explanations of the causal questions. The original paper is availiable at: 



The following provides an instance from the e-CARE dataset:

| Key          | Value                                                        |
| ------------ | ------------------------------------------------------------ |
| Premise      | Tom holds a copper block by hand and heats it on fire.       |
| Ask-for      | Effect                                                       |
| Hypothesis 1 | His fingers feel burnt immediately. (<font color=Green>&#10004;</font>) |
| Hypothesis 2 | The copper block keeps the same. (<font color=Red>&#x2716;</font>) |
| Explanation  | Copper is a good thermal conductor.                          |



Each instance of the e-CARE dataset is constituted by two components: 

* Multiple-choice Causal Reasoning Question

A multiple-choice causal reasoning question consists of a premise and two hypotheses, and one of the hypotheses can form a valid causal fact with the premise; 

* Conceptual Explanation

A conceptual explanation is about the essential condition that enables the existence of the causal fact. For example, as the above instance shows, the explanation points out the nature of copper that Copper is a good thermal conductor, so that holding copper on fire will make fingers feel burnt immediately. 


## Basic Statistics

* The question type distribution

|        | Train  | Test  |  Dev  | Total |
| :----: | :----: | :---: | :---: | :---: |
| Cause  | 7,617  | 2,176 | 1,088 | 10881 |
| Effect | 7,311  | 2,088 | 1,044 | 10443 |
| Total  | 14,928 | 4,264 | 2,132 | 21324 |

* The label distribution

|      | Train | Test | Dev  |
| ---- | ----- | ---- | ---- |
| 0    | 7463  | 2132 | 1066 |
| 1    | 7465  | 2132 | 1066 |

* Average length of cause, effect, wrong hypothesis and conceptual explanation

|                        | Overall | Train | Test | Dev  |
| ---------------------- | ------- | ----- | ---- | ---- |
| Conceptual Explanation | 7.63    | 7.62  | 7.60 | 7.77 |
| Cause                  | 8.51    | 8.51  | 8.47 | 8.56 |
| Effect                 | 8.34    | 8.33  | 8.38 | 8.31 |
| Distractor Option      | 8.14    | 8.14  | 8.10 | 8.21 |

* Number of conceptual explanations

| Overall | Train | Test | Dev  |
| ------- | ----- | ---- | ---- |
| 13048   | 10491 | 3814 | 2012 |



## Dataset, Training & Evaluation

To train and evaluate the model, the complete training and dev set can be downloaded at:
The causal question of testing set is provided in: 
To evaluate the model performance on test set, you need upload the results to:



## Baseline Results

On this basis, we introduce two tasks:

+ Causal Reasoning Task
  The causal reasoning task is a multiple-choice task: given a premise event, one needs to choose a more plausible hypothesis from two candidates, so that the premise and the correct hypothesis can form into a valid causal fact.

  | Model            | Dev   | Test  |
  | ---------------- | ----- | ----- |
  | Bart-base        | 73.03 | 71.65 |
  | Bert-base-cased  | 75.47 | 75.38 |
  | RoBERTa-base     | 70.64 | 70.73 |
  | XLNet-base-cased | 75.61 | 74.58 |
  | ALBERT           | 73.97 | 74.60 |
  | GPT              | 67.59 | 68.15 |
  | GPT-2            | 70.36 | 69.51 |



+ Explanation Generation Task
  It requires the model to generate a free-text-formed explanation for a given causal fact (composed of a premise and the corresponding correct hypothesis).

| Model      | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Rouge-1 | Rouge-2 | Rouge-l | PPL   |
| ---------- | ------ | ------ | ------ | ------ | ------- | ------- | ------- | ----- |
| GPT-2      | 55.17  | 33.29  | 23.00  | 18.79  | 33.17   | 10.23   | 32.05   | 6.87  |
| RNN        | 43.25  | 18.20  | 6.76   | 4.16   | 20.79   | 2.20    | 20.85   | 33.84 |
| Multi-Task | 56.32  | 35.96  | 26.47  | 22.36  | 35.70   | 12.57   | 34.88   | 6.64  |




## Future Direction

### Serve as a Causality Knowledge Base

Causal knowledge is critical for various NLP applications. The causality knowledge provided by e-CARE can be used as a resource to boost model performance on other causal-related tasks. 

We have made exploration by applying transfer learning by first finetuning a BERT model on e-CARE, then adapting the e-CARE-enhanced model (denoted as BERT$_\text{E}$) on a causal extraction task EventStoryLine [1], two causal reasoning tasks BECauSE 2.0 [2] and COPA [3], as well as a commonsense reasoning dataset CommonsenseQA[4]. The results are shown in the table below. We observe that the additional training process on e-CARE can consistently increase the model performance on all four tasks. This indicates the potential of e-CARE in providing necessary causality information for promoting causal-related tasks in multiple domains.

### Abductive Reasoning

Previous literature concluded the explanation generation process as an **abductive reasoning** process, and highlighted the importance of the abdutive explanation generation, as it may interact with the causal reasoning process to promote the understanding of causal mechanism, and increase the efficiency and reliability of causal reasoning.

For example, as the following figure shows, one may have an observation that $C_1$: *adding rock into hydrochloric acid* caused $E_1$: *rock dissolved*. Through abductive reasoning, one may come up with a conceptual explanation for the observation that *acid is corrosive*. After that, one can confirm or rectify the explanation by experiments, or resorting to external references. In this way, new ideas about causality can be involved for understanding the observed causal fact. Then if the explanation is confirmed, it can be further utilized to support the causal reasoning process by helping to explain and validate other related causal facts, such as $C_2$: *adding rust into sulphuric acid* may lead to $E_2$: *rust dissolved*.  This analysis highlights the pivotal role of conceptual explanation in learning and inferring causality and the e-CARE dataset to provide causal explanations and support future research towards stronger human-like causal reasoning systems. 
<style>table {margin: auto;}</style>


## References
[1] Caselli T, Vossen P. The event storyline corpus: A new benchmark for causal and temporal relation extraction[C]//Proceedings of the Events and Stories in the News Workshop. 2017: 77-86.

[2] Dunietz J, Levin L, Carbonell J G. The BECauSE corpus 2.0: Annotating causality and overlapping relations[C]//Proceedings of the 11th Linguistic Annotation Workshop. 2017: 95-104.

[3] Roemmele M, Bejan C A, Gordon A S. Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning[C]//AAAI spring symposium: logical formalizations of commonsense reasoning. 2011: 90-95.

[4] Talmor A, Herzig J, Lourie N, et al. CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge[C]//Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019: 4149-4158.
