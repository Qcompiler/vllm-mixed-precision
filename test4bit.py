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
# model="Baichuan2-7b"
model="Qwen2.5-7B-Instruct"
# model="Llama-2-7b"
os.system("rm /home/dataset/quant4/%s/*.safetensors"%(model))
llm = LLM(model="/home/dataset/quant4/%s"%(model), 
        quantization="MixQ4bit",trust_remote_code=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")