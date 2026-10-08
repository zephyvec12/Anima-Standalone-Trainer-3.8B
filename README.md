# Anima Standalone Trainer — 3.8B Edition

Anima **3.8B v1.1 LoRA training** through the original Standalone Jobs Web UI.
This fork adds the native 52-block backend, frozen dual text encoders, bundled
Semantic Connector v2 support, and per-prompt checkpoint comparisons in Samples.

Forked from [gazingstars123/Anima-Standalone-Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer).
The native backend comes from [GumGum10/sd-scripts](https://github.com/GumGum10/sd-scripts)
at a pinned commit. [Original README](README.upstream.md) · [3.8B details](docs/anima38.md).

## Quick start

Requires Python **3.11+** (3.12 recommended), Node.js, and an NVIDIA CUDA GPU.

```text
git clone https://github.com/zephyvec12/Anima-Standalone-Trainer-3.8B.git
cd Anima-Standalone-Trainer-3.8B
python tools/setup_anima38.py
cd training-ui
npm install
node server.js
```

Open `http://localhost:3000`.
The installer creates **venv38** with Torch 2.9.1 / CUDA 12.8 and downloads a
fixed native-backend revision with the required patches. Existing training data,
model files, job settings, and prompts are not copied into the repository.

1. Download the four files from [Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B):
   `Anima-3.8B-v1.1.safetensors`, `qwen_3_06b_base.safetensors`,
   `qwen35_4b.safetensors`, and `qwen_image_vae.safetensors`.
2. In **Global Settings → Anima 3.8B v1.1**, set their local paths. The server uses
   `venv38` for native 3.8B jobs when the global Venv Path is blank; otherwise set
   Venv Path to your 3.8B environment.
3. Create a job. In **Dataset**, add every intended subset and its repeats.
4. In **Prompts**, enter your own trial prompts. In **Training**, choose the
   checkpoint and sampling interval, then click **Train**.
5. Open **Samples**: every prompt has its own group containing that prompt's
   samples across checkpoints. Each image shows its training step, and its
   filename identifies its checkpoint. Choose **All** to display every version.

The initial style preset is batch **2**, accumulation **1**, rank/alpha **32/32**,
LR **2e-5**, bf16, 1536 buckets, and checkpoint/trial saves every **250** steps.
It is a starting point; your job settings remain authoritative.

**Scope:** single-GPU 3.8B LoRA training and saved training trials. The 3.8B adapter
currently does not provide Manual Generate, full finetuning, or multi-GPU
training. See [configuration, sample formats, and validation](docs/anima38.md).

## 中文说明

这是 Anima 3.8B v1.1 的训练适配版，保留原项目的 Jobs、Dataset、Prompts、Samples
和 TensorBoard 页面。使用两个冻结的文本编码器，并从 v1.1 模型中提取其自带的
Semantic Connector v2；训练对象为 DiT LoRA。

Samples 按 Prompt 1、Prompt 2 等分别展示各个 checkpoint 的结果，图片下方标明步数，
文件名对应保存的 checkpoint。兼容原版和旧版云端的样图文件名。训练器不向你的提示词
添加额外单词，也不创建额外提示词变体。训练数据、私人提示词和模型权重不在发布内容中。

当前已在单张 RTX 4090 上完成实际训练和采样验证；Windows 安装入口已提供，尚未完成
一次全新的 Windows GPU 训练验证。详细参数、安装方式和限制见 [3.8B 文档](docs/anima38.md)。

## License

The original Standalone Trainer license is retained in [LICENSE.md](LICENSE.md).
The downloaded native backend retains its upstream license. Model weights have
their own upstream licenses; they are downloaded separately and are not bundled.
