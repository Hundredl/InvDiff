from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
import torch.nn as nn
import torch
from typing import Dict, Any, Optional, Tuple, Union
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
import copy
class EpsilonWithDelta(UNet2DConditionModel):
    def __init__(self, pretrain_unet=None, delta=None, delta_ratio=0.5, delta_init0=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if pretrain_unet is not None:
        for name in list(self._modules.keys()):
            delattr(self, name)
            if name == 'config':
                continue
        print(f'self.config: {self.config}')
        
        self.pretrain_unet = pretrain_unet
        self.delta = delta
        self.delta_ratio = delta_ratio

        if delta is None:
            config = pretrain_unet.config
            print("---------------------------")
            print(config)
            print("---------------------------")
            # create a unet model, which is the same as the pretrain_unet, copy from the pretrain_unet
            self.delta = copy.deepcopy(pretrain_unet)
            
        if pretrain_unet is not None:
            # freeze the pretrain_unet
            for param in self.pretrain_unet.parameters():
                param.requires_grad = False


            all_params = sum(p.numel() for p in self.pretrain_unet.parameters())
            print(f"pretrain_unet's parameters: {all_params}count, {all_params/1e6}M")
        all_params = sum(p.numel() for p in self.delta.parameters())
        print(f"delta's parameters: {all_params}count, {all_params/1e6}M")
        # print all the parameters
        print("EpsilonWithDelta's parameters:")
        if delta_init0:
            self.delta.conv_out.weight.data.fill_(0)
            self.delta.conv_out.bias.data.fill_(0)

    
            
    def forward(
        self,
        *args,
        **kwargs,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        pretrain_output = self.pretrain_unet(
            *args,
            **kwargs,
        )
        # use the delta to get the delta_output
        delta_output = self.delta(
            *args,
            **kwargs,
        )
        # return the result of the delta_output and pretrain
        if self.delta_ratio <= -1:
            out_put = delta_output[0] + pretrain_output[0]
        elif self.delta_ratio > 1:
            out_put = self.delta_ratio * delta_output[0] + pretrain_output[0]
        else:    
            out_put = self.delta_ratio * delta_output[0] + (1-self.delta_ratio) * pretrain_output[0]
        
        return (out_put,)

