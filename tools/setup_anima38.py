"""Install the pinned native backend and optional isolated CUDA environment."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BACKEND_COMMIT = '2d76db07396f2a36cc1dbd484d32d010cc3bcb46'
BACKEND_URL = f'https://codeload.github.com/GumGum10/sd-scripts/zip/{BACKEND_COMMIT}'


def patch_backend(backend):
    model_path = backend / 'library/anima_utils.py'
    text = model_path.read_text(encoding='utf-8')
    before = '    if fp8_scaled:\n        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)'
    after = '''    # The v1.1 bundle includes connector tensors. Load the exact extracted
    # connector separately, retaining strict validation of the DiT itself.
    connector_keys = [key for key in sd if key.startswith("anima_v2_connector.")]
    for key in connector_keys:
        del sd[key]
    if connector_keys:
        logger.info("Separated %d bundled v2 connector tensors", len(connector_keys))

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)'''
    if after not in text:
        if text.count(before) != 1:
            raise RuntimeError('Pinned backend model loader does not match the expected patch.')
        model_path.write_text(text.replace(before, after, 1), encoding='utf-8')
    train_path = backend / 'train_network.py'
    text = train_path.read_text(encoding='utf-8')
    call = '''                    self.sample_images(
                        accelerator, args, None, global_step, accelerator.device, vae, tokenizers, text_encoder, unet
                    )
                    progress_bar.unpause()
'''
    if '# anima38-ui: checkpoint before sampling' not in text:
        start = text.index('                    global_step += 1')
        call_start = text.index(call, start)
        text = text[:call_start] + text[call_start:].replace(call, '', 1)
        insert = text.index('                    optimizer_train_fn()', call_start)
        text = text[:insert] + '                    # anima38-ui: checkpoint before sampling\n' + call + text[insert:]
        train_path.write_text(text, encoding='utf-8')


def install_backend(destination):
    marker = destination / 'ANIMA38_BACKEND.json'
    if marker.is_file():
        if json.loads(marker.read_text())['commit'] != BACKEND_COMMIT:
            raise RuntimeError(f'Another backend already exists at {destination}. Select a new --backend-dir.')
        patch_backend(destination)
        return
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f'Backend directory is not empty: {destination}')
    print(f'Downloading native backend at {BACKEND_COMMIT}', flush=True)
    with urllib.request.urlopen(BACKEND_URL, timeout=120) as response:
        archive = response.read()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for member in bundle.infolist():
            relative = Path(*Path(member.filename).parts[1:])
            if not relative.parts:
                continue
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError('Backend archive contains an invalid path.')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(member))
    patch_backend(destination)
    marker.write_text(json.dumps({'repository': 'GumGum10/sd-scripts', 'commit': BACKEND_COMMIT,
                                 'patches': ['bundled-v11-connector', 'checkpoint-before-sampling']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-dir', type=Path, default=ROOT / 'backends/anima38/sd-scripts')
    parser.add_argument('--venv', type=Path, default=ROOT / 'venv38')
    parser.add_argument('--backend-only', action='store_true', help='Download and patch source without installing packages.')
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error('Python 3.11+ is required; Python 3.12 is recommended.')
    install_backend(args.backend_dir.resolve())
    if not args.backend_only:
        env = args.venv.resolve()
        if not env.exists():
            venv.EnvBuilder(with_pip=True).create(env)
        python = env / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
        subprocess.run([str(python), '-m', 'pip', 'install', 'torch==2.9.1', 'torchvision==0.24.1',
                        '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-anima38.txt')], check=True)
        print(f'Anima 3.8B venv: {env}')
    print(f'Native backend ready: {args.backend_dir.resolve()}')


if __name__ == '__main__':
    main()
