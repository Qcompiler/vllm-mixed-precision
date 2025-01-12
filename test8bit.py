import sys

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    # "Hello, my name is",
    "The president of the United States is",
    # "The capital of France is",
    # "What is a Wiki? Wiki wiki sign outside Honolulu International Airport. (Image courtesy of A. Barataz)The name Wiki was chosen by Ward Cunningham -- the creator of the first Wiki. It is a shortened form of wiki-wiki, the Hawaiian word for quick.A wiki is a web site that is generally editable by anyone with a computer, a web browser, and an internet connection. Wikis use a quick and easy syntax to allow users to apply formatting to text and create links between pages. This simple formatting syntax means that authors no longer need to learn the complexities of HTML to create content on the web.The main strength of a wiki is that it gives people the ability to work collaboratively on the same document. The only software you need is an Internet browser. Consequently, wikis are used for a variety of purposes. If you make a mistake, it's easy to revert back to an earlier version of the document.",
    # "Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G. A. True, True B. False, False C. True, False D. False, True Answer: B  Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements. A. True, True B. False, False C. True, False D. False, True Answer: C "
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
import os

# please use Qcompiler to quant model !

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--quant", type=str, default="0")
parser.add_argument("--model_dir", type=str, default="/home/dataset/mixqdata")

parser.add_argument("--model", type=str, default="Llama-2-7b")
args = parser.parse_args()
if args.quant == "mixq8":
    model_dir = os.path.join(args.model_dir ,args.model)
    quantization = "MixQ8bit"
    print("use mixed 8bit quant")
    os.system("rm -rf " + model_dir + "/model.safetensors")
elif args.quant == "mixq4":
    model_dir =  os.path.join(args.model_dir ,args.model)
    quantization = "MixQ4bit"    
    os.system("rm -rf " + model_dir + "/model.safetensors")
elif args.quant == "awq":
    model_dir =  os.path.join(args.model_dir ,args.model)
    quantization = "AWQ" 
elif args.quant == "gptq":
    model_dir =  os.path.join(args.model_dir ,args.model)
    quantization = "GPTQ"    
else:
    model_dir =   os.path.join(args.model_dir ,args.model)
    quantization = None
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=128)


llm = LLM(model=model_dir, trust_remote_code=True,
        quantization=quantization, max_num_seqs=128)
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")