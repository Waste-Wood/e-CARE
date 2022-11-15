python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "bert-base-cased" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "bert" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "BCE" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "bert-base-cased" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "bert" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 5e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "BCE" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "bert-base-cased" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "bert" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-6 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "BCE" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "bert-base-cased" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "bert" \
  --gpu "0" \
  --batch_size 32 \
  --cuda True\
  --epochs 2 \
  --lr 5e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "BCE" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "bert-base-cased" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "bert" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "CrossEntropy" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "bert-base-uncased" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "bert" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "BCE" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "bert-base-uncased" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "bert" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "CrossEntropy" \


python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "roberta-base" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "roberta" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "BCE" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "roberta-base" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "roberta" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "CrossEntropy" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "roberta-base" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "roberta" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "CrossEntropy" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "xlnet-base-cased" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "xlnet" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "BCE" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "albert-base-v2" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "albert" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "BCE" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "gpt2" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "gpt2" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "BCE" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "openai-gpt" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "gpt" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "BCE" \

python3 train_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "facebook/bart-base" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "dev.jsonl" \
  --dev "dev.jsonl" \
  --test "dev.jsonl" \
  --model_name "bart" \
  --gpu "0" \
  --batch_size 64 \
  --cuda True\
  --epochs 2 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 \
  --loss_func "BCE" \