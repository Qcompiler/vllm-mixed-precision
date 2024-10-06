import sys

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
import os

# please use Qcompiler to quant model !

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--quant", type=int, default=0)

args = parser.parse_args()
if args.quant == 8:
    model_dir = "/home/cyd/dataset/quant8"
    quantization = "MixQ8bit"
if args.quant == 4:
    model_dir = "/home/cyd/dataset/quant4mix"
    quantization = "MixQ4bit"    
else:
    model_dir = "/home/cyd/dataset"
    quantization = None
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
model_dir = os.path.join(model_dir,"Qwen2-7B-Instruct")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=128)

llm = LLM(model=model_dir, trust_remote_code=True,
        quantization=quantization, max_num_seqs=128)
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")