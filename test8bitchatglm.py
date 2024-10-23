import sys

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "Please write CUDA code to show how to use cutlass to compute  fp16 GEMM:",
    "Please write a long poem: "
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=128)

# Create an LLM.
import os

# please use Qcompiler to quant model !

os.system("rm /dev/shm/tmp/quant8/chatglm/chatglm2-6b/*.safetensors")
# 删除量化后的tonken chatglm https://github.com/THUDM/ChatGLM3/issues/152
os.system("rm /dev/shm/tmp/quant8/chatglm/chatglm2-6b/tokenizer*")
os.system("cp -r /dev/shm/tmp/chatglm/chatglm2-6b/tokenizer*   /dev/shm/tmp/quant8/chatglm/chatglm2-6b/")

llm = LLM(model="/dev/shm/tmp/quant8/chatglm/chatglm2-6b", trust_remote_code=True,
        quantization="MixQ8bit", max_num_seqs=128)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")