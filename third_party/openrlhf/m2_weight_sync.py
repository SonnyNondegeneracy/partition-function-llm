"""Effective HF tensors for Qwen3.5 LoRA -> ordinary vLLM weight loading.

Only standard, unquantized LoRA with ZeRO <= 2 and TP=1 is supported.
The actor's base and adapter tensors are never merged/unmerged in place.
HF names are retained: vLLM's own Qwen3.5 loader owns HF-to-vLLM mapping.
"""
import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer

# Non-language towers that vLLM (language_model_only) does not load. Qwen3.5 uses
# 'model.visual.'; the Gemma4 names were added for the NMI follow-up runs.
MULTIMODAL_PREFIXES = ('model.visual.', 'model.vision_tower.', 'model.audio_tower.',
                       'model.embed_vision.', 'model.embed_audio.', 'model.multi_modal_projector.')


def weight_entries(model):
    if not isinstance(model, PeftModel):
        raise TypeError('M2 requires a PEFT LoRA model')
    for cfg in model.peft_config.values():
        if cfg.use_dora or cfg.bias != 'none' or cfg.modules_to_save:
            raise ValueError('Only standard bias-free LoRA is supported')
    base = model.get_base_model()
    modules = dict(base.named_modules())
    entries = []
    for name, param in base.named_parameters():
        if name.startswith(MULTIMODAL_PREFIXES) or '.lora_' in name:
            continue
        if not (name.startswith('model.language_model.') or name.startswith('lm_head.')):
            raise ValueError(f'Unexpected Qwen3.5/Gemma4 text parameter: {name}')
        layer = None
        if '.base_layer.' in name:
            parent, leaf = name.split('.base_layer.', 1)
            layer = modules[parent]
            if not isinstance(layer, LoraLayer) or layer.merged or layer.disable_adapters:
                raise ValueError(f'Unsupported LoRA state at {parent}')
            if leaf != 'weight':
                raise ValueError(f'Unsupported LoRA parameter: {name}')
            name = parent + '.' + leaf
        entries.append((name, param, layer))
    if not any(layer is not None for _, _, layer in entries):
        raise ValueError('No language LoRA weights selected')
    return entries


@torch.no_grad()
def effective_weight(param, layer):
    if layer is None:
        return param.detach()
    # Exactly PEFT's standard merge order/arithmetic, but into a fresh buffer.
    weight = param.detach().clone()
    for adapter in layer.active_adapters:
        if adapter in layer.lora_A:
            weight.add_(layer.get_delta_weight(adapter).to(weight.dtype))
    if not torch.isfinite(weight).all():
        raise ValueError('Nonfinite effective weight')
    return weight
