from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.quantization.mixq import MixQLinearMethod

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
    def from_config(cls, config: Dict[str, Any]) -> "MixQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        print("mix------weight")
        print(weight_bits)
        print(group_size)
        return cls(weight_bits, group_size)

    def get_quant_method(
            self, layer: torch.nn.Module,prefix: str
            ) -> Optional["MixQLinear4bitMethod"]:
        if isinstance(layer, LinearBase):
            #print("--get_quant_method---")
            #print(layer.name)
            if layer.name is not None and "down" in layer.name:
                #print("use 8bit!")
                return MixQLinearMethod(self, weight_only = True)
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

 
        qweight = Parameter(
            torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        qzeros = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        scales = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": 0,
            "output_dim": 1,
        })

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(qweight, extra_weight_attrs)
        set_weight_attrs(scales, extra_weight_attrs)
        set_weight_attrs(qzeros, extra_weight_attrs)
        # self.quant_config.group_size = 128

        # output_size_per_partition = sum(output_partition_sizes)
        # if output_size_per_partition % 8 != 0:
        #     raise ValueError(
        #         "The output size is not aligned with the quantized "
        #         "weight shape. This can be caused by too large "
        #         "tensor parallel size.")

 


    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        

        shape = x.shape[:-1] + (layer.weight.shape[0], )


        inputs = x.reshape(-1, x.shape[-1])
        M =  inputs.shape[0]
        N = layer.weight.shape[0]
        K = inputs.shape[1]
 
        if M > 32:

            y1 = mixlib.mixgemmforward4bit(M,N,K,
                                x,
                                layer.weight, 
                                layer.scale_col,
                                layer.weight_cache, #新增fp weight 
                                layer.ind, #新增fp ind 
                                layer.qweight, #新增 int4 weight int32
                                layer.scales)
        else:
            qweight = layer.qweight
            scales = layer.scales
            qzeros = layer.qzeros
            pack_factor = 8
            out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
            reshaped_x = x.reshape(-1, x.shape[-1])

            # num_tokens >= threshold
            FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256

            if FP16_MATMUL_HEURISTIC_CONDITION:
                out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
                out = torch.matmul(reshaped_x, out)
            else:
                out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros,
                                pack_factor)
            if bias is not None:
                out.add_(bias)
            return out.reshape(out_shape)
        if layer.bias is not None:
            y1 += layer.bias

        
        return y1.reshape(shape)
        
        
