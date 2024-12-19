from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

def FindOutliers(Activation, sigma = None):

    if sigma is None:
        sigma = 10
    
    tmp = torch.unique(torch.where((  Activation.abs() > sigma ))[1])
    return tmp.to(torch.int32)
class MixQConfig(QuantizationConfig):
    """Config class for MixQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int
    ) -> None:
        self.weight_bits = weight_bits
        # self.group_size = group_size
        

    def __repr__(self) -> str:
        return (f"MixQConfig(weight_bits={self.weight_bits}, ")

    def get_name(self) -> str:
        return "MixQ"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        # The MixQ kernel only supports Turing or newer GPUs.
        return 75

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
        # group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        #print("mix------weight")
        #print(weight_bits)
        #print(group_size)
        return cls(weight_bits)

    def get_quant_method(
            self, layer: torch.nn.Module, prefix: str) -> Optional["MixQLinearMethod"]:
        if isinstance(layer, LinearBase):
            # print("--get_quant_method---")
            # print(layer.prefix)
            if layer.prefix is not None and "down" in layer.prefix:
                return MixQLinearMethod(self, weight_only = True)
            return MixQLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]

import mixlib
from EETQ import w8_a16_gemm
class MixQLinearMethod(LinearMethodBase):


    def __init__(self, quant_config: MixQConfig, weight_only = False):
        self.quant_config = quant_config
        self.weight_only = weight_only
        


    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        # print(self.quant_config.group_size)
        # print(input_size_per_partition)
        # if input_size_per_partition % self.quant_config.group_size != 0:
        #     raise ValueError(
        #         "The input size is not aligned with the quantized "
        #         "weight shape. This can be caused by too large "
        #         "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)

        #print(input_size_per_partition)
        #print(output_size_per_partition)

  
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
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


        q_weight = Parameter(
            torch.empty(
                input_size_per_partition,
                output_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            q_weight, {
                "input_dim": 0,
                "output_dim": 1,
                 
            })
        layer.register_parameter("q_weight", q_weight)
        set_weight_attrs(q_weight, extra_weight_attrs)


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



        q_scale_col = Parameter(
            torch.empty(
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(q_scale_col, {
            "output_dim": 0,
        })
        layer.register_parameter("q_scale_col", q_scale_col)
        set_weight_attrs(q_scale_col, extra_weight_attrs)

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
        layer.capture = 0
        layer.n_outliers = 128
        
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        

        shape = x.shape[:-1] + (layer.weight.shape[0], )


        inputs = x.reshape(-1, x.shape[-1])
        M =  inputs.shape[0]
        N = layer.weight.shape[0]
        K = layer.weight.shape[1]

        apply_capture = True
        if not layer.capture and apply_capture:


            layer.capture = 1
            local_ind = FindOutliers(inputs)
            
            layer.n_outliers = len(local_ind)      
    
            # if  layer.n_outliers:
            #     layer.ind.data = local_ind
            #     layer.weight_cache.data = layer.weight.data[:, local_ind] *  layer.scale_col.reshape((N, 1))
            #     layer.weight_cache.data = layer.weight_cache.data.to(torch.float16)

        if self.weight_only is True:

            y1 =  w8_a16_gemm(inputs,layer.q_weight,layer.q_scale_col)
            # print('-------------')
            # print(inputs)
            # print(layer.q_weight)
            # print(layer.q_scale_col)
            # print(y1)
            # print('-------------')
            # if self.weight_only is True:

            #     exit()
        else:
            # for compute bound 
            if M > 64:

                if layer.n_outliers == 0:
                    #print("no outliers!")

        
                    y1 = mixlib.mixgemmforward_direct(M,N,K,
                                        x,
                                        layer.weight, 
                                        layer.scale_col)


                else:

                    #print("call  weight act",M,N,K)
                    y1 = mixlib.mixgemmforward(M,N,K,
                                        x,
                                        layer.weight, 
                                        layer.scale_col,
                                        layer.weight_cache, #新增fp weight 
                                        layer.ind, #新增fp ind 
                                        layer.q_weight, #新增 int4 weight int32
                                        layer.q_scale_col)
            else:
                
                #print("call  weight only",M,N,K)
                y1 =  w8_a16_gemm(inputs,layer.q_weight,layer.q_scale_col)



        if layer.bias is not None:
            y1 += layer.bias

        
        # #print(self.ind.shape[0])
        # # print("grand is ")
        # # print(y1)
        # # print("EETQ is ")
        # # print(y2)
        
        # err = torch.norm(y1 - y2)
        # print(layer.name)
        # if   layer.q_weight.shape[1] == 11008*2:
        #     #torch.save(layer.q_weight,"eetq.pt")
        #     #torch.save(inputs,"activation.pt")
        #     #torch.save(layer.q_scale_col,"q_scale_col.pt")
        #     #print(layer.weight.shape)
            
        #     #print(layer.q_weight.shape)
        #     print(layer.q_weight)
        #     #print(layer.q_weight[1,0:4096])
        #     print(err)
        #     exit(0)
            
        #exit(0)
        return y1.reshape(shape)
        
        
