#!/bin/bash

# hyper-parameter
policy_model_path="gpt-4o-2024-08-06"
reward_model_path="LLaMA-3-8B-SFR-Iterative-DPO-R"
reference_model_path="Llama-3-8b-Instruct"
autoais_model_path="t5_xxl_true_nli_mixture"
retriever="gtr"
query_sample="3"
answer_sample="1"
retrieval_topk="3"
num_simulation="30"
max_num_layers="5"
expand_probability="0.4"
c_param="0.2"
value_threshold="1.0"
seed="42"
reflexion_threshold="10"
gpu_ids="2,3"

data_path="../data/asqa_gtr_top100.json"
prompt_path="prompts/asqa.yaml"
start_idx="0"
end_idx="1000"
log_path="../logging/asqa_gpt-4o-2024-08-06.txt"
save_path="../output/asqa/asqa_gpt-4o-2024-08-06_0_1000.json"

export DPR_WIKI_TSV=$PWD/psgs_w100.tsv
export GTR_EMB=$PWD/gtr_wikipedia_index.pkl
export RETRIEVER="gtr-t5-xxl"

python main.py --policy_model_path ${policy_model_path} --reward_model_path ${reward_model_path} --reference_model_path ${reference_model_path} \
       --autoais_model_path ${autoais_model_path} --query_sample ${query_sample} --answer_sample ${answer_sample} --retrieval_topk ${retrieval_topk} \
       --num_simulation ${num_simulation} --max_num_layers ${max_num_layers} --expand_probability ${expand_probability} \
       --c_param ${c_param} --value_threshold ${value_threshold} --data_path ${data_path} --prompt_path ${prompt_path} \
       --start_idx ${start_idx} --end_idx ${end_idx} --log_path ${log_path} --save_path ${save_path} --seed ${seed} --gpu_ids ${gpu_ids}
