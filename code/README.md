## Code for ACL' 2022 paper: e-CARE: a New Dataset for Exploring Explainable Causal Reasoning


#### &#x1F603;requirements

* python >= 3.6
* torch >= 1.3.1
* transformers >= 4.4.1
* scikit-learn >= 0.19.1
* rouge >= 1.0.0 
* OpenNMT-py >= 2.1.2


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#### &#x1F3c3;Running Causal Reasoning Task

* fine-tuning with **GPT-2**

  ```shell
  python3 gpt2_discriminate.py \
    --data_dir "../data/Causal_Reasoning/" \
    --model_dir "../../huggingface_transformers/gpt2/" \
    --save_dir "./output/saved_model" \
    --log_dir "./output/log" \
    --train "train.jsonl" \
    --dev "dev.jsonl" \
    --test "test.jsonl" \
    --model_name "gpt2" \
    --cuda True \
    --gpu "0" \
    --batch_size 64 \
    --epochs 100 \
    --evaluation_step 200 \
    --lr 1e-5 \
    --set_seed True \
    --seed 338 \
    --patient 3 \
  ```

* fine-tuning with **BART-base**

  ```shell
  python3 train_discriminate.py \
    --data_dir "../data/Causal_Reasoning/" \
    --model_dir "../../huggingface_transformers/bart-base/" \
    --save_dir "./output/saved_model" \
    --log_dir "./output/log" \
    --train "train.jsonl" \
    --dev "dev.jsonl" \
    --test "test.jsonl" \
    --model_name "bart" \
    --gpu "0" \
    --batch_size 64 \
    --cuda True\
    --epochs 100 \
    --evaluation_step 200 \
    --lr 1e-5 \
    --set_seed True \
    --seed 338 \
    --patient 3 \
    --loss_func "BCE" \
  ```

* fine-tuning with **BERT-base**

  ```shell
  python3 train_discriminate.py \
    --data_dir "../data/Causal_Reasoning/" \
    --model_dir "../../huggingface_transformers/bert-base-cased/" \
    --save_dir "./output/saved_model" \
    --log_dir "./output/log" \
    --train "train.jsonl" \
    --dev "dev.jsonl" \
    --test "test.jsonl" \
    --model_name "bert" \
    --gpu "0" \
    --batch_size 64 \
    --cuda True\
    --epochs 100 \
    --evaluation_step 250 \
    --lr 1e-5 \
    --set_seed True \
    --seed 338 \
    --patient 3 \
    --loss_func "BCE" \
  ```

* fine-tuning with **RoBERTa-base**

  ```shell
  python3 train_discriminate.py \
    --data_dir "../data/Causal_Reasoning/" \
    --model_dir "../../huggingface_transformers/roberta-base/" \
    --save_dir "./output/saved_model" \
    --log_dir "./output/log" \
    --train "train.jsonl" \
    --dev "dev.jsonl" \
    --test "test.jsonl" \
    --model_name "roberta" \
    --cuda True \
    --gpu "0" \
    --batch_size 64 \
    --epochs 100 \
    --evaluation_step 200 \
    --lr 1e-5 \
    --set_seed True \
    --seed 338 \
    --patient 3 \
    --loss_func "BCE" \
  ```

* fine-tuning with **XLNet-base**

  ```shell
  python3 train_discriminate.py \
    --data_dir "../data/Causal_Reasoning/" \
    --model_dir "../../huggingface_transformers/xlnet-base-cased/" \
    --save_dir "./output/saved_model" \
    --log_dir "./output/log" \
    --train "train.jsonl" \
    --dev "dev.jsonl" \
    --test "test.jsonl" \
    --model_name "xlnet" \
    --cuda True \
    --gpu "0" \
    --batch_size 64 \
    --epochs 100 \
    --evaluation_step 200 \
    --lr 1e-5 \
    --set_seed True \
    --seed 338 \
    --patient 3 \
    --loss_func "BCE" \
  ```

* fine-tuning with **ALBERT-base**

  ```shell
  python3 train_discriminate.py \
    --data_dir "../data/Causal_Reasoning/" \
    --model_dir "../../huggingface_transformers/albert-base-v2/" \
    --save_dir "./output/saved_model" \
    --log_dir "./output/log" \
    --train "train.jsonl" \
    --dev "dev.jsonl" \
    --test "test.jsonl" \
    --model_name "albert" \
    --cuda True \
    --gpu "0" \
    --batch_size 64 \
    --epochs 100 \
    --evaluation_step 200 \
    --lr 1e-5 \
    --set_seed True \
    --seed 338 \
    --patient 3 \
    --loss_func "BCE" \
  ```

* fine-tuning with **GPT**

  ```shell
  python3 train_discriminate.py \
    --data_dir "../data/Causal_Reasoning/" \
    --model_dir "../../huggingface_transformers/gpt/" \
    --save_dir "./output/saved_model" \
    --log_dir "./output/log" \
    --train "train.jsonl" \
    --dev "dev.jsonl" \
    --test "test.jsonl" \
    --model_name "gpt" \
    --cuda True \
    --gpu "0" \
    --batch_size 64 \
    --epochs 100 \
    --evaluation_step 200 \
    --lr 1e-5 \
    --set_seed True \
    --seed 338 \
    --patient 3 \
    --loss_func "BCE" \
  ```

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#### &#x1F3C4;Running  Explanation Generation Task

* fine-tuning with **GPT-2**

  ```shell
  python3 gpt2_generate.py \
  	--data_dir '../data/Explanation_Generation/' \
  	--model_dir '../../huggingface_transformers/gpt2/' \
  	--save_dir './output/saved_model' \
  	--log_dir './output/log' \
  	--train 'train.jsonl' \
  	--dev 'dev.jsonl' \
  	--test 'test.jsonl' \
  	--model_name 'gpt2' \
  	--cuda True \
  	--gpu '0' \
  	--batch_size 32 \
  	--epochs 100 \
  	--evaluation_step 200 \
  	--lr 1e-5 \
  	--seed 1024 \
  	--patient 5 \
  	--length 22 \
  ```

* training with **RNN**

  ```shell
  cd ./model/rnn
  
  # build vocab
  onmt_build_vocab -config config.yaml
  
  # training rnn-based seq2seq model
  onmt_train -config config.yaml
  
  # prediction
  onmt_translate -config generate_config.yaml
  
  # Bleu & Rouge
  python3 metrics.py
  ```

  PS: data format can refer to [OpenNMT-py](https://opennmt.net/OpenNMT-py/quickstart.html)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#### &#x1F46B;Running Multi-task Training (discriminate-generate)

```shell
python gpt2_multi_task.py \
  --data_dir "../data/" \
  --model_dir "../../huggingface_transformers/gpt2/" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "train.jsonl" \
  --dev "dev.jsonl" \
  --test "test.jsonl" \
  --model_name "gpt2" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 100 \
  --evaluation_step 200 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 5 \
  --length 22 \
  --alpha 0.9 \
  --beam_size 5 \
  --no_repeat_ngram_size 3 \
  --repetition_penalty 1.5 \
  --do_sample True \
  --mode "discriminate_generate" \
```


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#### &#x1F525;Adversarial Filtering & Gradient Attack

* running adversarial filtering

```shell
python3 adversarial_filtering.py
```

PS: The published e-CARE dataset has been processed by adversarial filtering.



* running gradient attack

```shell
python3 gpt2_discriminate.py \
  --data_dir "./data/final_data/data/" \
  --model_dir "../../huggingface_transformers/gpt2/" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "train.jsonl" \
  --dev "dev.jsonl" \
  --test "test.jsonl" \
  --model_name "gpt2" \
  --cuda True \
  --gpu "0" \
  --batch_size 64 \
  --epochs 100 \
  --evaluation_step 200 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
```

