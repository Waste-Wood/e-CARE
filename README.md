# e-CARE: a New Dataset for Exploring Explainable Causal Reasoning

## Brief Introduction
Understanding causality has vital importance for various Natural Language Processing (NLP) applications. Beyond the labeled instances, conceptual explanations of the causality can provide deep understanding of the causal fact to facilitate the causal reasoning process. We present a human-annotated explainable CAusal REasoning dataset (e-CARE), which contains over 20K causal reasoning questions, together with natural language formed explanations of the causal questions. The original paper is availiable at: 

The following provides an instance from the e-CARE dataset:

|Premise:| Tom holds a copper block by hand and heats it on fire.|
|Ask-for:|Effect|
|Hypothesis 1:| His fingers feel burnt immediately. (√)|
|Hypothesis 2:| The copper block keeps the same. (×)|
|Explanation: Copper is a good thermal conductor.|

Each instance of the e-CARE dataset is constituted by two components: 
(1) a multiple-choice causal reasoning question, composed of a premise and two hypotheses, and one of the hypotheses can form a valid causal fact with the premise; 
(2) a conceptual explanation about the essential condition that enables the existence of the causal fact. For example, as the above instance shows, the explanation points out the nature of copper that Copper is a good thermal conductor, so that holding copper on fire will make fingers feel burnt immediately. 

## Basic Statistics

Subset Train Test Dev
Cause 7617 2176 1088
Effect 7311 2088 1044

## Basic Statistics

To train and evaluate the model, the complete training and dev set can be downloaded at:
The causal question of testing set is provided in: 
To evaluate the model performance on test set, you need upload the results to:

## Baseline Results

On this basis, we introduce two tasks:

+ Causal Reasoning Task
The causal reasoning task is a multiple-choice task: given a premise event, one needs to choose a more plausible hypothesis from two candidates, so that the premise and the correct hypothesis can form into a valid causal fact.

+ Explanation Generation Task
It requires the model to generate a free-text-formed explanation for a given causal fact (composed of a premise and the corresponding correct hypothesis).


## Future Direction

### Serve as a Causality Knowledge Base

Causal knowledge is critical for various NLP applications. The causality knowledge provided by e-CARE can be used as a resource to boost model performance on other causal-related tasks. 

We have made exploration by applying transfer learning by first finetuning a BERT model on e-CARE, then adapting the e-CARE-enhanced model (denoted as BERT$_\text{E}$) on a causal extraction task EventStoryLine [1], two causal reasoning tasks BECauSE 2.0 [2] and COPA [3], as well as a commonsense reasoning dataset CommonsenseQA[4]. The results are shown in the table below. We observe that the additional training process on e-CARE can consistently increase the model performance on all four tasks. This indicates the potential of e-CARE in providing necessary causality information for promoting causal-related tasks in multiple domains.

### Abductive Reasoning

Previous literature concluded the explanation generation process as an **abductive reasoning** process, and highlighted the importance of the abdutive explanation generation, as it may interact with the causal reasoning process to promote the understanding of causal mechanism, and increase the efficiency and reliability of causal reasoning.

For example, as the following figure shows, one may have an observation that $C_1$: *adding rock into hydrochloric acid* caused $E_1$: *rock dissolved*. Through abductive reasoning, one may come up with a conceptual explanation for the observation that *acid is corrosive*. After that, one can confirm or rectify the explanation by experiments, or resorting to external references. In this way, new ideas about causality can be involved for understanding the observed causal fact. Then if the explanation is confirmed, it can be further utilized to support the causal reasoning process by helping to explain and validate other related causal facts, such as $C_2$: *adding rust into sulphuric acid* may lead to $E_2$: *rust dissolved*.  This analysis highlights the pivotal role of conceptual explanation in learning and inferring causality and the e-CARE dataset to provide causal explanations and support future research towards stronger human-like causal reasoning systems. 
