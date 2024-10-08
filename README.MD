# Mixed-precision Inference In VLLM

The mixed-precision inference is used for accelerating prefill steps and for enhancing throughput of LLM.

## Setup
Please quantized the model by the QComplier (https://github.com/Qcompiler/QComplier)

```
git clone  git@github.com:Qcompiler/QComplier.git

```

Build the kernel by

```
cd EETQ
python setup.py install
cd quantkernel
python setup.py install
```




Please install the vllm by
```
pip install vllm==0.6.2
```


Please install the mixed-precision source code by
```
git clone git@github.com:Qcompiler/vllm-mixed-precision.git
```

And copy the ".so" from the vllm project

```
cp -r $PYTHON_PATH/lib/python3.11/site-packages/vllm/*.so  vllm-mixed-precision/*
```

Delete the vllm==0.6.2
```
pip uninstall vllm
```


## Quant the 8bit and 4bit mixed-precsion LLMs

```
cd QComplier/src
bash quant.sh
```

## Runing 8-bit mixed-preiciosn infernce in vllm

```
cd examples/
python test8bit.p --quant 8
```


<!-- ## Runing 4-bit mixed-preiciosn infernce in vllm

We support for 4-bit mixed-precision quantizion  -->

## Acknowledgement

[1] QUIK (https://arxiv.org/pdf/2310.09259)

[2] VLLM (https://github.com/vllm-project/vllm)