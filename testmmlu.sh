cmd="srun -N 1 --gres=gpu:4090:1 "
set -ex
 
base=/home/chenyidong/data/mixqdata
models=( "chatglm2-6b"   "falcon-7b"   "vicuna-7b"  ) 
models=(   "Llama-2-7b" "chatglm2-6b"   "falcon-7b"   "vicuna-7b" ) 
models=(     "glm-4-9b-chat"   ) 
models=(    "glm-4-9b-chat"    ) 
for model in "${models[@]}"
    do
    ${cmd} python mmlu.py --model_type fp16 \
    --data_dir  ${base}/data/data \
    --hf_model_dir  ${base}/${model}  \
    --tokenizer_dir  ${base}/${model}

done

# models=(    "falcon-7b"   ) 
# for model in "${models[@]}"
#     do
#     ${cmd} python mmlu.py --model_type mixq8 \
#     --data_dir  ${base}/data/data \
#     --hf_model_dir  ${base}/quant8/${model}  \
#     --tokenizer_dir  ${base}/${model}

# done



# models=(    "glm-4-9b-gptq-4bit"   ) 
# for model in "${models[@]}"
#     do
#     ${cmd} python mmlu.py --model_type GPTQ \
#     --data_dir  ${base}/data/data \
#     --hf_model_dir  ${base}/${model}  \
#     --tokenizer_dir  ${base}/${model}

# done


