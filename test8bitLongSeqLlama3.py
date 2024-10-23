import sys

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "你好，请写一首中文诗，字数不低于1000字.",
    "如何使用cutlass 在H100上实现 FP8的 GEMM,请写一段代码.",
    "请介绍一下美国的历史，字数不低于1000字.",
    "请写一份政府工作报告.",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens = 64)

# Create an LLM.
import os

# please use Qcompiler to quant model !
model="Llama3-Chinese_v2"
os.system("rm /dev/shm/tmp/quant8/%s/*.safetensors"%(model))
llm = LLM(model="/dev/shm/tmp/quant8/%s"%(model), 
        quantization="MixQ8bit",trust_remote_code=True)
    
#llm = LLM(model="/dev/shm/tmp/%s"%(model),  trust_remote_code=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")