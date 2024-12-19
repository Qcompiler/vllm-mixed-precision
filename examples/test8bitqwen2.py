import sys

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Please write a poem:" ,
    "请写一首中文诗歌:"

]
# Create a sampling params object.

max_tokens = 1024
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens, min_tokens=max_tokens)

# Create an LLM.
import os

# please use Qcompiler to quant model !

os.system("rm /dev/shm/tmp/quant8/Qwen2-7B-Instruct/*.safetensors")
# 删除量化后的tonken chatglm https://github.com/THUDM/ChatGLM3/issues/152
#os.system("rm /dev/shm/tmp/quant8/chatglm/chatglm2-6b/tokenizer*")
#os.system("cp -r /dev/shm/tmp/Qwen2-7B-Instruct/tokenizer*   /dev/shm/tmp/quant8/Qwen2-7B-Instruct/")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--quant", type=int, default=0)

args = parser.parse_args()

if args.quant:
    path = "/home/cyd/dataset/quant8"
    quantization = "MixQ8bit"
else:
    path = "/home/cyd/dataset"
    quantization = None
llm = LLM(model=(os.path.join(path,"Qwen2-7B-Instruct")), trust_remote_code=True,
           quantization = quantization)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
import time
start = time.time()
outputs = llm.generate(prompts, sampling_params)
end = time.time()
duration =  end - start 
duration = max_tokens / duration 
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("%.2f tokens/seconds \n"%(duration))
