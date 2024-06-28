# adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py

import torch

from diffusers.models.attention import BasicTransformerBlock
from torch.nn.parallel import DistributedDataParallel

from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

class ReferenceAttentionControl():
    
    def __init__(self, 
                 unet,
                 mode="write",
                 reference_attn=True,
                 fusion_blocks="midup",
                 do_classifier_free_guidance=False,
                 style_fidelity=1.,
                 repeats=1,
                 ) -> None:
        self.unet = unet
        assert mode in ["read", "write"], "mode must be write or read"
        assert fusion_blocks in ["midup", "full"], "fusion type should be midup or full"
        assert 0<=style_fidelity<=1, "make sure style_fidelity between 0 and 1"
        self.reference_attn = reference_attn
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode, 
            do_classifier_free_guidance=do_classifier_free_guidance,
            style_fidelity=style_fidelity,
            repeats=repeats,
        )
    
    def register_reference_hooks(
            self, 
            mode, 
            do_classifier_free_guidance=False,
            style_fidelity=1.,
            repeats=1,
        ):
        MODE = mode

        # https://github.com/huggingface/diffusers/blob/main/examples/community/stable_diffusion_reference.py
        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
        ):
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    self.bank.append(norm_hidden_states)
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                elif MODE == "read" and len(self.bank)==0:
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                elif MODE == "read":
                    ref_hidden_states = torch.cat(self.bank, dim=1).to(device=norm_hidden_states.device,dtype=norm_hidden_states.dtype).repeat_interleave(repeats,dim=0)
                    
                    # vanilla attention with reference
                    num_repeat = norm_hidden_states.shape[0]//ref_hidden_states.shape[0]
                    ref_states = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=torch.cat([norm_hidden_states,ref_hidden_states.repeat(num_repeat,1,1)], dim=1),
                        # attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                    ref_states += hidden_states

                    attn_output_vanilla = None
                    if do_classifier_free_guidance: # or norm_hidden_states.shape[0]>ref_hidden_states.shape[0]
                        attn_output_vanilla = self.attn1(norm_hidden_states, 
                                                encoder_hidden_states=norm_hidden_states,
                                                attention_mask=attention_mask,
                                                **cross_attention_kwargs,)
                        attn_output_vanilla += hidden_states
                        mid = attn_output_vanilla.shape[0]//2
                        attn_output_vanilla[-mid:] = ref_states[-mid:]
                        if style_fidelity<1:
                            attn_output_vanilla[:mid] = (1-style_fidelity)*ref_states[:mid] + style_fidelity*attn_output_vanilla[:mid]
                    else:
                        attn_output_vanilla = ref_states
                    attn_output = attn_output_vanilla

                    self.bank.clear()
                    hidden_states = attn_output.clone()
                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                        )
                        hidden_states = (
                            self.attn2(
                                norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                            )
                            + hidden_states
                        )

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
                    return hidden_states

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states


        if self.reference_attn:
            if self.fusion_blocks == "midup":
                if isinstance(self.unet, DistributedDataParallel):
                    attn_modules = [module for module in (torch_dfs(self.unet.module.mid_block)+torch_dfs(self.unet.module.up_blocks)) if isinstance(module, BasicTransformerBlock)]
                else:
                    attn_modules = [module for module in (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks)) if isinstance(module, BasicTransformerBlock)]
            elif self.fusion_blocks == "full":
                attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]            
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))
    

    def share_bank(self, writer, dtype=torch.float16, num_repeat=1):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                if isinstance(self.unet, DistributedDataParallel):
                    reader_attn_modules = [module for module in (torch_dfs(self.unet.module.mid_block)+torch_dfs(self.unet.module.up_blocks)) if isinstance(module, BasicTransformerBlock)]
                else:
                    reader_attn_modules = [module for module in (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks)) if isinstance(module, BasicTransformerBlock)]
                if isinstance(writer.unet, DistributedDataParallel):
                    writer_attn_modules = [module for module in (torch_dfs(writer.unet.module.mid_block)+torch_dfs(writer.unet.module.up_blocks)) if isinstance(module, BasicTransformerBlock)]
                else:
                    writer_attn_modules = [module for module in (torch_dfs(writer.unet.mid_block)+torch_dfs(writer.unet.up_blocks)) if isinstance(module, BasicTransformerBlock)]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
                writer_attn_modules = [module for module in torch_dfs(writer.unet) if isinstance(module, BasicTransformerBlock)]
            reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])    
            writer_attn_modules = sorted(writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                # r.bank = [v.clone().repeat_interleave(num_repeat,dim=0).to(dtype) for v in w.bank]
                # w.bank.clear()
                r.bank = w.bank

    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                if isinstance(self.unet, DistributedDataParallel):
                    reader_attn_modules = [module for module in (torch_dfs(self.unet.module.mid_block)+torch_dfs(self.unet.module.up_blocks)) if isinstance(module, BasicTransformerBlock)]
                else:
                    reader_attn_modules = [module for module in (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks)) if isinstance(module, BasicTransformerBlock)]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
            reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
            for r in reader_attn_modules:
                r.bank.clear()

    def reset(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                if isinstance(self.unet, DistributedDataParallel):
                    attn_modules = [module for module in (torch_dfs(self.unet.module.mid_block)+torch_dfs(self.unet.module.up_blocks)) if isinstance(module, BasicTransformerBlock)]
                else:
                    attn_modules = [module for module in (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks)) if isinstance(module, BasicTransformerBlock)]
            elif self.fusion_blocks == "full":
                attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]            
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module.forward = module._original_inner_forward
                module.bank = []