from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.quantization.mixq import MixQLinearMethod

import mixgemm
class MixQ4bitConfig(QuantizationConfig):
    """Config class for MixQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.pack_factor = 8
        

    def __repr__(self) -> str:
        return (f"MixQ4bitConfig(weight_bits={self.weight_bits}, ")

    def get_name(self) -> str:
        return "MixQ"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        # The MixQ kernel only supports Turing or newer GPUs.
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-MixQ
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-MixQ
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any])  :
        weight_bits = 4
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        # print("mix------weight")
        # print(weight_bits)
        # print(group_size)
        return cls(weight_bits, group_size)

    def get_quant_method(
            self, layer: torch.nn.Module,prefix: str
            ) -> Optional["MixQLinear4bitMethod"]:
        if isinstance(layer, LinearBase):
            #print("--get_quant_method---")
            print(layer.prefix)
            if layer.prefix is not None and "down" in layer.prefix:
                print("use 8bit!")
                return MixQLinearMethod(self)
            return MixQLinear4bitMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]

import mixlib
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
class MixQLinear4bitMethod(LinearMethodBase):


    def __init__(self, quant_config ):
        self.quant_config = quant_config
        self.debug = False


    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)

        print(input_size_per_partition)
        print(output_partition_sizes)
        print("4bit")
        

        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight, {
                "input_dim": 1,
                "output_dim": 0,
                 
            })
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)




        
        fp_features_num = 128
        weight_cache = Parameter(
            torch.empty(
                output_size_per_partition,
                fp_features_num,
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight_cache, {
                "input_dim": 1,
                "output_dim": 0,
                 
            })
        layer.register_parameter("weight_cache", weight_cache)
        set_weight_attrs(weight_cache, extra_weight_attrs)
        

        ind = Parameter(
            torch.empty(
                fp_features_num,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("ind", ind)
        set_weight_attrs(ind, extra_weight_attrs)




        scale_col = Parameter(
            torch.empty(
                1,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scale_col, {
            "input_dim": 0,
            "output_dim": 1,
        })
        layer.register_parameter("scale_col", scale_col)
        set_weight_attrs(scale_col, extra_weight_attrs)

 
        q_weight = Parameter(
            torch.empty(
                output_size_per_partition ,
                input_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            q_weight, {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        # q_scale_col = Parameter(
        #     torch.empty(
        #         input_size_per_partition // self.quant_config.group_size,
        #         output_size_per_partition // self.quant_config.pack_factor,
        #         dtype=torch.int32,
        #     ),
        #     requires_grad=False,
        # )
        # set_weight_attrs(
        #     q_scale_col, {
        #         "input_dim": 0,
        #         "output_dim": 1,
        #         "packed_dim": 1,
        #         "pack_factor": self.quant_config.pack_factor,
        #     })
        q_scale_col = Parameter(
            torch.empty(
                1,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(q_scale_col, {
            "input_dim": 0,
            "output_dim": 1,
        })

        layer.register_parameter("q_weight", q_weight)
        layer.register_parameter("q_scale_col", q_scale_col)
        # layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(q_weight, extra_weight_attrs)
        set_weight_attrs(q_scale_col, extra_weight_attrs)
        # set_weight_attrs(qzeros, extra_weight_attrs)
        # self.quant_config.group_size = 128

        # output_size_per_partition = sum(output_partition_sizes)
        # if output_size_per_partition % 8 != 0:
        #     raise ValueError(
        #         "The output size is not aligned with the quantized "
        #         "weight shape. This can be caused by too large "
        #         "tensor parallel size.")
        layer.init = False
        layer.weight_cache2 = torch.clone(weight_cache.data)
        layer.out = torch.zeros((1, output_size_per_partition), dtype=torch.half, device=weight_cache.data.device)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        

        shape = x.shape[:-1] + (layer.weight.shape[0], )


        inputs = torch.clone(x.reshape(-1, x.shape[-1]))
 
        M =  inputs.shape[0]
        N = layer.weight.shape[0]
        K = inputs.shape[1]

        # print("call 4 bit mix")
        # print(x.shape,end="")
        # print(layer.weight.shape)
        # if  torch.isnan(torch.abs(torch.sum(inputs[0:1024]))) :
            
            
        #     print("mixq 4 torch.isnan(torch.abs(torch.sum(inputs[0:1024])))")
        #     print(torch.sum(inputs[0:1024]))
        if  layer.init is False:
            layer.weight_cache2 = layer.weight_cache.data / layer.q_scale_col.T
            layer.q_scale_col.data  = layer.q_scale_col.data.to(torch.float32)
            layer.init = True                

        #     exit()
        if M > 1:
            
            y1 = mixlib.mixgemmforward4bit(M,N,K,
                                inputs,
                                layer.weight, 
                                layer.scale_col,
                                layer.weight_cache, #新增fp weight 
                                layer.ind, #新增fp ind 
                                layer.q_weight, #新增 int4 weight int32
                                layer.q_scale_col)
        else:
            qweight = layer.q_weight
            mixgemm.gemv_int4_fp16_mix(M, N, K, inputs, layer.q_weight,
                                   layer.out, 64, 4, 
                                   layer.q_scale_col,
                                   layer.weight_cache2, layer.ind, 
                                   128)
            if bias is not None:
                layer.out.add_(bias)
            return layer.out.reshape(shape)
        if layer.bias is not None:
            y1 += layer.bias

        
        # print(y1)
        return y1.reshape(shape)
        
        
