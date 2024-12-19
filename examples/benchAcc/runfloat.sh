

#CMD="srun  -p twills -A h100 --gres=gpu:h100:1 --export=ALL python"
#CMD="python "
pip uninstall bitsandbytes

CMD=" python "
export http_proxy=127.0.0.1:8892 
export https_proxy=127.0.0.1:8892
set -x

model_path='/dev/shm/tmp'
models=(    "Llama-2-7b" )

for batch in    512 
    do

            
            
        # models=(  "Llama-2-7b" "Baichuan2-7b" "Baichuan2-13b" "Llama-65b"  "Llama-2-70b" "Aquila2-7b" "Aquila2-34b" falcon-7b "falcon-40b" "Mistral-7b")  
        # models=(    "opt-30b" )
        data_types=( "fp16" )
        for data_type in "${data_types[@]}"
            do
            for model in "${models[@]}"
                do
                echo ${model}          
                CUDA_VISIBLE_DEVICES=0     ${CMD} evalppl.py  --model_type ${data_type} --model_path  \
                ${model_path}/${model} \
                --quant_file ${model_path}/${model} \
                --n_ctx $batch --n_batch $batch  --eval_accuracy True
            done
        done



     
done
