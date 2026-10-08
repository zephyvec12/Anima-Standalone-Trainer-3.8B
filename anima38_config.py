"""Translate Standalone Jobs settings to the pinned native 3.8B backend."""
from __future__ import annotations

from pathlib import Path


def flatten(config):
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result.update(value)
        else:
            result[key] = value
    return result


def build_native_config(merged, allowed, connector_path):
    ui = flatten(merged)
    native = {key: value for key, value in ui.items() if key in allowed}
    for source, target in [('dit_path', 'pretrained_model_name_or_path'),
                           ('qwen3_path', 'qwen3'), ('vae_path', 'vae')]:
        value = ui.get(source) or ui.get(target)
        if not value:
            raise ValueError(f'Missing model path: {source}. Set it in Global Settings → Anima 3.8B.')
        native[target] = value
    # Compatible with existing cloud jobs whose old UI knows only qwen3_path.
    native['qwen35'] = ui.get('qwen35_path') or ui.get('qwen35') or str(
        Path(native['qwen3']).with_name('qwen35_4b.safetensors'))
    native['semantic_connector'] = str(connector_path)
    native['network_module'] = 'networks.lora_anima'
    native['network_train_unet_only'] = True
    user_network_args = [arg for arg in ui.get('network_args', [])
                         if not arg.startswith('train_llm_adapter=')]
    native['network_args'] = [*user_network_args, 'train_llm_adapter=false']
    native['text_encoder_lr'] = 0.0
    native['unet_lr'] = native['learning_rate']
    # Preserve the optimizer and all user arguments, including explicit betas.
    optimizer_args = list(native.get('optimizer_args') or [])
    defaults = [('betas', '(0.9,0.99)'), ('eps', '1e-8')] if 'adam' in native.get('optimizer_type', '').lower() else []
    for key, value in defaults:
        if not any(arg.split('=', 1)[0] == key for arg in optimizer_args):
            optimizer_args.append(f'{key}={value}')
    native['optimizer_args'] = optimizer_args
    if native.get('lr_scheduler') == 'constant' and native.get('lr_warmup_steps'):
        native['lr_scheduler'] = 'constant_with_warmup'
    native['timestep_sampling'] = 'sigmoid'
    native['weighting_scheme'] = 'none'
    native['attn_mode'] = 'torch'
    native['qwen_image_vae_2d'] = True
    native['cache_latents'] = native['cache_latents_to_disk'] = True
    native['cache_text_encoder_outputs'] = native['cache_text_encoder_outputs_to_disk'] = True
    native.setdefault('text_encoder_batch_size', 4)
    native['sample_at_first'] = False
    # Base Anima calls this qwen3_token_length; the native backend calls it max_token_length.
    native['qwen3_max_token_length'] = ui.get('qwen3_max_token_length', ui.get('qwen3_token_length', 512))
    native['qwen35_max_token_length'] = ui.get('qwen35_max_token_length', 512)
    native['t5_token_length'] = ui.get('t5_token_length', 512)
    return native
