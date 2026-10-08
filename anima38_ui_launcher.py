"""Run the author 3.8B trainer from the existing Standalone Trainer Jobs UI."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parent
INITIAL_CWD = Path.cwd()
WORK = Path(os.environ.get('ANIMA38_CACHE_DIR', ROOT / '.anima38'))
REPO = Path(os.environ.get('ANIMA38_BACKEND_DIR', ROOT / 'backends/anima38/sd-scripts'))
if not (REPO / 'anima_38b_train_network.py').is_file():
    raise RuntimeError('Native backend is missing. Run: python tools/setup_anima38.py')
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import torch
from PIL import Image, PngImagePlugin
from safetensors import safe_open
from safetensors.torch import save_file
import toml

from anima_38b_train_network import Anima38BNetworkTrainer, setup_parser
from library import args as args_util, checkpoint_io, sampling, strategy_base
from library.device_utils import clean_memory_on_device
from anima38_config import build_native_config, flatten

log = logging.getLogger('anima38_ui')


def extract_bundled_connector(bundle: str) -> str:
    import hashlib
    source = Path(bundle).resolve()
    stat = source.stat()
    identity = hashlib.sha256(f'{source}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:16]
    destination = WORK / 'connectors' / f'{source.stem}-{identity}.safetensors'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return str(destination)
    prefix = 'net.anima_v2_connector.'
    with safe_open(bundle, framework='pt', device='cpu') as checkpoint:
        metadata = checkpoint.metadata() or {}
        connector_metadata = {
            k[len('anima_v2_adapter_'):]: v
            for k, v in metadata.items() if k.startswith('anima_v2_adapter_')
        }
        connector_metadata['architecture'] = (
            'anima_qwen35_quality_anchored_semantic_connector_v2'
        )
        tensors = {
            k[len(prefix):]: checkpoint.get_tensor(k)
            for k in checkpoint.keys() if k.startswith(prefix)
        }
    if not tensors:
        raise ValueError('The selected model does not contain the Anima v1.1 Semantic Connector v2.')
    print('Extracting the connector bundled with this v1.1 model:', len(tensors),
          'tensors', flush=True)
    save_file(tensors, str(destination), metadata=connector_metadata)
    return str(destination)


class UITrainer(Anima38BNetworkTrainer):
    def __init__(self):
        super().__init__()
        self.trials = []
        self.trial_conditions = {}

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype
    ):
        original_vae_device = vae.device
        vae.to('cpu')
        clean_memory_on_device(accelerator.device)
        for encoder in text_encoders:
            encoder.to(accelerator.device)
        with accelerator.autocast():
            dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)
        tokenize = strategy_base.TokenizeStrategy.get_strategy()
        encode = strategy_base.TextEncodingStrategy.get_strategy()
        self.trials = sampling.load_prompts(args.sample_prompts) if args.sample_prompts else []
        print('Caching trial conditioning for', len(self.trials), 'prompts', flush=True)
        with torch.no_grad(), accelerator.autocast():
            for prompt in self.trials:
                for text in (prompt.get('prompt', ''), prompt.get('negative_prompt', '')):
                    if text not in self.trial_conditions:
                        result = encode.encode_tokens(tokenize, text_encoders, tokenize.tokenize(text))
                        self.trial_conditions[text] = [t.detach().cpu() for t in result]
        accelerator.wait_for_everyone()
        for encoder in text_encoders:
            encoder.to('cpu')
        vae.to(original_vae_device)
        clean_memory_on_device(accelerator.device)

    def sample_images(
        self, accelerator, args, epoch, global_step, device, vae, tokenizer,
        text_encoder, unet
    ):
        if not self.trials or global_step == 0:
            return
        if epoch is None:
            interval = args.sample_every_n_steps
            if not interval or global_step % interval:
                return
        elif not args.sample_every_n_epochs or epoch % args.sample_every_n_epochs:
            return
        model = accelerator.unwrap_model(unet)
        was_training = model.training
        model.eval()
        model.switch_block_swap_for_inference()
        output = Path(args.output_dir) / 'sample' / f'step_{global_step:06d}'
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_name = checkpoint_io.get_step_ckpt_name(
            args, '.' + args.save_model_as, global_step)
        if (Path(args.output_dir) / checkpoint_name).is_file():
            sample_prefix = Path(checkpoint_name).stem
        else:
            checkpoint_name = None
            sample_prefix = f'{args.output_name}__preview_step_{global_step:06d}'
        rows = []
        print('Generating saved trials at optimizer step', global_step, flush=True)
        try:
            with torch.no_grad(), accelerator.autocast(), torch.random.fork_rng(devices=[device]):
                for index, prompt in enumerate(self.trials):
                    seed_value = prompt.get('seed')
                    seed = int(69411 + index if seed_value is None else seed_value)
                    height = int(prompt.get('height', 1216)) // 16 * 16
                    width = int(prompt.get('width', 832)) // 16 * 16
                    steps = int(prompt.get('sample_steps', 30))
                    cfg = float(prompt.get('scale', 4.0))
                    latents = self.sample_one(model, prompt, seed, height, width, steps, cfg, device)
                    old_vae_device = vae.device
                    vae.to(device)
                    try:
                        decoded = vae.decode_to_pixels(latents).float()
                    finally:
                        vae.to(old_vae_device)
                    pixels = ((decoded + 1) / 2).clamp(0, 1)[0]
                    if pixels.ndim == 4:
                        pixels = pixels[:, 0]
                    pixels = (pixels.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    file = output / f'{sample_prefix}__prompt_{index+1:02d}_seed_{seed}.png'
                    # Match the original trainer's PNG metadata, using exact user text.
                    metadata = PngImagePlugin.PngInfo()
                    parameters = prompt.get('prompt', '')
                    if prompt.get('negative_prompt'):
                        parameters += '\nNegative prompt: ' + prompt['negative_prompt']
                    parameters += (f'\nSteps: {steps}, Sampler: Euler, CFG scale: {cfg}, '
                                   f'Seed: {seed}, Size: {width}x{height}, Training step: {global_step}')
                    metadata.add_text('parameters', parameters)
                    metadata.add_text('training_step', str(global_step))
                    metadata.add_text('prompt_index', str(index))
                    if checkpoint_name:
                        metadata.add_text('checkpoint', checkpoint_name)
                    Image.fromarray(pixels).save(file, pnginfo=metadata)
                    rows.append({'file': file.name, 'step': global_step, 'prompt_index': index,
                                 'seed': seed, 'checkpoint': checkpoint_name})
                    del decoded, pixels, latents
                    print('Trial saved:', global_step, index + 1, flush=True)
            (output / 'manifest.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        finally:
            model.train(was_training)
            model.switch_block_swap_for_training()
            model.prepare_block_swap_before_forward()
            clean_memory_on_device(device)

    def sample_one(self, model, prompt, seed, height, width, steps, cfg, device):
        dtype = next(model.parameters()).dtype
        conditions = []
        for text in (prompt.get('prompt', ''), prompt.get('negative_prompt', '')):
            cached = self.trial_conditions[text]
            native = cached[0].to(device, dtype=dtype)
            native_mask = cached[1].to(device)
            t5_ids = cached[2].to(device, dtype=torch.long)
            t5_mask = cached[3].to(device)
            semantic = [t.to(device, dtype=dtype) for t in cached[4:-1]]
            semantic_mask = cached[-1].to(device)
            conditions.append((native, native_mask, t5_ids, t5_mask, semantic, semantic_mask))
        generator = torch.Generator(device='cpu').manual_seed(seed)
        x = torch.randn((1,16,1,height//8,width//8), generator=generator).to(device, dtype=dtype)
        sigmas = torch.linspace(1, 0, steps+1, device=device, dtype=torch.float32)
        sigmas = 3 * sigmas / (1 + 2 * sigmas)
        padding = torch.zeros((1,1,height//8,width//8), device=device, dtype=dtype)
        for step in range(steps):
            t = sigmas[step:step+1].to(dtype)
            predictions = []
            for native, native_mask, t5_ids, t5_mask, semantic, semantic_mask in conditions:
                context = model.semantic_connector_v2(
                    native_source=native, target_input_ids=t5_ids,
                    semantic_hidden_states=semantic, target_attention_mask=t5_mask,
                    native_source_mask=native_mask, semantic_source_mask=semantic_mask,
                    timesteps=t)
                context = context.masked_fill(~t5_mask.bool().unsqueeze(-1), 0)
                model.prepare_block_swap_before_forward()
                predictions.append(model(x, t, context, padding_mask=padding).float())
            velocity = predictions[1] + cfg * (predictions[0] - predictions[1])
            x = (x.float() + (sigmas[step+1] - sigmas[step]) * velocity).to(dtype)
        return x


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config_file', required=True)
    cli.add_argument('--check-config', action='store_true', help='Translate settings without loading models or starting training.')
    options = cli.parse_args()
    config_file = Path(options.config_file)
    if not config_file.is_absolute():
        config_file = INITIAL_CWD / config_file
    merged = tomllib.loads(config_file.read_text(encoding='utf-8'))
    parser = setup_parser()
    allowed = {action.dest for action in parser._actions}
    model = flatten(merged)
    bundle = model.get('dit_path') or model.get('pretrained_model_name_or_path')
    if options.check_config:
        connector = WORK / 'connectors/check-config.safetensors'
    else:
        connector = extract_bundled_connector(bundle)
    native = build_native_config(merged, allowed, connector)
    destination = config_file.with_name('_native38_config.toml')
    if options.check_config:
        print(json.dumps({key: native.get(key) for key in [
            'network_dim', 'network_alpha', 'learning_rate', 'max_train_steps',
            'gradient_accumulation_steps', 'optimizer_type', 'blocks_to_swap',
            'qwen3_max_token_length', 'qwen35_max_token_length', 't5_token_length']}, indent=2))
        return
    destination.write_text(toml.dumps(native), encoding='utf-8')
    native_cli_args = ['--qwen35', native['qwen35'],
        '--semantic_connector', native['semantic_connector'], '--config_file', str(destination)]
    args = parser.parse_args(native_cli_args)
    args_util.verify_command_line_training_args(args)
    # The backend reparses argv after TOML loading. Keep required arguments on both passes.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *native_cli_args]
        args = args_util.read_config_from_file(args, parser)
    finally:
        sys.argv = original_argv
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        raise ValueError('The 3.8B Jobs adapter currently supports single-GPU training.')
    os.chdir(REPO)
    print('Starting native Anima 3.8B v1.1 LoRA training through Jobs UI.', flush=True)
    print(f'accumulation={args.gradient_accumulation_steps} dim={args.network_dim} '
          f'alpha={args.network_alpha} lr={args.learning_rate} steps={args.max_train_steps} '
          f'text_cache_batch={args.text_encoder_batch_size}', flush=True)
    UITrainer().train(args)


if __name__ == '__main__':
    main()
