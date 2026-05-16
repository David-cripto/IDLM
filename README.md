# IDLM: Inverse-distilled Diffusion Language Models

<p align="center">
  <b>Few-step text generation from distilled diffusion.</b><br>
  Research code for <i>IDLM: Inverse-distilled Diffusion Language Models</i> (ICML 2026).
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.19066">
    <img src="https://img.shields.io/badge/arXiv-2602.19066-b31b1b.svg" alt="arXiv">
  </a>
  
  <a href="https://huggingface.co/kekchpek">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffcc4d.svg" alt="Hugging Face Models">
  </a>
  
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  </a>
</p>


## What is IDLM?

Diffusion Language Models can generate high-quality text, but their iterative reverse-diffusion sampling makes inference slow. **IDLM** speeds them up by distilling a pretrained many-step diffusion language model into a **few-step generator**.

Instead of simply matching every teacher step, IDLM uses an inverse-distillation view for discrete token spaces:

1. start with a strong pretrained DLM teacher,
2. train a student generator to produce text in far fewer steps,
3. train a “fake” diffusion model on student samples,
4. update the student using the teacher–fake loss gap.

The paper reports **4×–64× fewer inference steps** while preserving the teacher model’s entropy and generative perplexity.

---

## Highlights

- **Fast diffusion-language generation** with 4, 8, 16, and 32 step sampling recipes.
- **Inverse distillation for discrete tokens**, including student/fake-model training logic.
- **Hydra-powered experiments** for easy configuration and reproducibility.
- **PyTorch Lightning training loop** with checkpointing, logging, and distributed training support.
- **Ready-to-run scripts** for MDLM, DUO, and DCD-style recipes.
- **Evaluation utilities** for NLL, BPD, perplexity, generative perplexity, and sample entropy.

---

## Repository layout

```text
IDLM/
├── configs/                 # Hydra configs: data, model, algo, strategy, callbacks, etc.
│   ├── algo/                # ar, mdlm, duo, duo_base, d3pm, sedd, distillation, ot-finetune
│   ├── data/                # OpenWebText configs
│   ├── model/               # tiny / small / medium model configs
│   ├── noise/               # diffusion noise schedules
│   └── config.yaml          # main experiment config
├── integral/                # precomputed tokenizer / integration assets
├── models/                  # DiT backbone, EMA utilities, attention tests
├── scripts/                 # training and generation recipes
├── algo.py                  # model families and IDLM distillation logic
├── dataloader.py            # tokenizers, datasets, dataloaders
├── main.py                  # Hydra + Lightning entry point
├── metrics.py               # perplexity, entropy, BPD, NLL metrics
├── trainer_base.py          # shared training / sampling base classes
├── utils.py                 # logging and helper utilities
├── requirements.txt         # environment note / dependency list
└── LICENSE
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/David-cripto/IDLM.git
cd IDLM
```

### 2. Create an environment

The provided configs target a CUDA setup. Adjust the Python, PyTorch, and CUDA versions to your machine if needed.

```bash
conda create -n idlm python=3.10 -y
conda activate idlm
```

Install the CUDA toolkit version used by the provided environment note:

```bash
conda install nvidia/label/cuda-12.4.0::cuda-toolkit -y
```

Install the main Python stack:

```bash
pip install -U pip

pip install \
  datasets==2.15.0 \
  einops==0.7.0 \
  fsspec \
  h5py==3.10.0 \
  hydra-core==1.3.2 \
  ipdb==0.13.13 \
  lightning==2.2.1 \
  notebook==7.1.1 \
  nvitop==1.3.2 \
  omegaconf==2.3.0 \
  packaging==23.2 \
  pandas==2.2.1 \
  rich==13.7.1 \
  seaborn==0.13.2 \
  scikit-learn==1.4.0 \
  transformers==4.38.2 \
  triton==2.2.0 \
  torch==2.3.1 \
  torchaudio==2.3.1 \
  torchmetrics==1.6.1 \
  torchvision==0.18.1 \
  wandb \
  timm \
  ocifs \
  hf_transfer \
  huggingface-hub
```

Optional, install FlashAttention after PyTorch is installed:

```bash
pip install flash_attn==2.7.4.post1 --no-build-isolation
```

> Note: `requirements.txt` currently acts more like a compact environment note than a standard pip requirements file. If you want `pip install -r requirements.txt` to work directly, place each package on its own uncommented line.

---

## Quick start

All experiments are driven by `main.py` and Hydra overrides.

The main modes are:

```text
mode=train        # train / distill
mode=ppl_eval     # validation perplexity evaluation
mode=sample_eval  # generate samples and compute sample metrics
```

---

## Train IDLM

The repository includes three training recipes.

### MDLM teacher → IDLM student

```bash
bash scripts/train_idlm_mdlm.sh
```

### DUO teacher → IDLM student

```bash
bash scripts/train_idlm_duo.sh
```

### DCD-style recipe

```bash
bash scripts/train_idlm_dcd.sh
```

These scripts use Hydra overrides for batch size, dataset, teacher checkpoint, algorithm, sampling steps, precision, logging name, and validation frequency. Use them as strong starting points, then tune the overrides for your compute budget.

---

## Generate samples

The generation scripts sweep over 4, 8, 16, and 32 sampling steps.

Before running them, set `eval.generated_samples_path` to a real JSON output path.

### MDLM checkpoint

```bash
mkdir -p samples

python -m main \
  mode=sample_eval \
  data=openwebtext-split \
  algo=mdlm \
  algo.backbone=hf_dit \
  eval.checkpoint_path=kekchpek/idlm-mdlm \
  sampling.steps=8 \
  sampling.num_sample_batches=10 \
  sampling.predictor=ancestral_cache \
  sampling.noise_removal=ancestral \
  eval.generated_samples_path=samples/idlm_mdlm_8steps.json
```

### DUO checkpoint

```bash
mkdir -p samples

python -m main \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo \
  algo.backbone=hf_dit \
  eval.checkpoint_path=kekchpek/idlm-duo \
  sampling.steps=8 \
  sampling.num_sample_batches=10 \
  sampling.noise_removal=greedy \
  eval.generated_samples_path=samples/idlm_duo_8steps.json
```

### Run the provided sweeps

```bash
bash scripts/generation_idlm_mdlm.sh
bash scripts/generation_idlm_duo.sh
bash scripts/generation_idlm_dcd.sh
```

Generated sample files contain:

```json
{
  "generative_ppl": 0.0,
  "entropy": 0.0,
  "generated_seqs": []
}
```

---

## Evaluate perplexity

```bash
python -m main \
  mode=ppl_eval \
  data=openwebtext-split \
  algo=mdlm \
  algo.backbone=hf_dit \
  eval.checkpoint_path=<path-or-hf-checkpoint> \
  loader.eval_batch_size=8
```

Replace `<path-or-hf-checkpoint>` with a local checkpoint path or a Hugging Face checkpoint identifier.

---

## Common Hydra overrides

| Override | Purpose |
| --- | --- |
| `mode=train` | Train or distill a model |
| `mode=sample_eval` | Generate samples |
| `mode=ppl_eval` | Run validation evaluation |
| `algo=mdlm` | Use MDLM |
| `algo=duo` | Use DUO |
| `algo=sedd` | Use SEDD |
| `algo=d3pm` | Use D3PM absorbing-state diffusion |
| `algo.backbone=hf_dit` | Use the Hugging Face DiT-style backbone |
| `data=openwebtext-split` | Use the OpenWebText split config |
| `model=small` | Use the small model config |
| `model.length=1024` | Set sequence length |
| `sampling.steps=8` | Set number of sampling steps |
| `eval.checkpoint_path=...` | Load a checkpoint for eval / sampling |
| `eval.generated_samples_path=...` | Save generated samples to JSON |
| `adversarial_distill.is_distill=True` | Enable IDLM-style distillation |
| `logger.name=...` | Name the TensorBoard run |

Example:

```bash
python -m main \
  mode=train \
  data=openwebtext-split \
  model=small \
  algo=mdlm \
  algo.backbone=hf_dit \
  adversarial_distill.is_distill=True \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  sampling.steps=32 \
  optim.lr=1e-6 \
  logger.name=idlm-mdlm
```

---

## Checkpoints referenced by the scripts

Training scripts reference teacher checkpoints such as:

```text
kuleshov-group/mdlm-owt
s-sahoo/duo
s-sahoo/duo-distilled
```

Generation scripts reference IDLM checkpoints such as:

```text
kekchpek/idlm-mdlm
kekchpek/idlm-duo
kekchpek/idlm-dcd
```

Make sure your environment can access the required checkpoints before running training or generation.

---

## Outputs

By default, Hydra writes experiment outputs under:

```text
outputs/<dataset>/<date>/<time>/
```

TensorBoard logs are written under:

```text
tb_logs/
```

Checkpoints are written according to the checkpointing config in `configs/config.yaml`.

---

## Tips

- Reduce `loader.batch_size`, `loader.eval_batch_size`, or `model.length` if you run out of GPU memory.
- Install FlashAttention only after PyTorch is installed.
- For generation, always set `eval.generated_samples_path` to a file path, for example `samples/run.json`.
- Use `trainer.devices=1` for a single-GPU run.
- Use `trainer.limit_val_batches=0.01` for quick validation checks during debugging.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{li2026idlm,
  title={IDLM: Inverse-distilled Diffusion Language Models},
  author={Li, David and Gushchin, Nikita and Abulkhanov, Dmitry and Moulines, Eric and Oseledets, Ivan and Panov, Maxim and Korotin, Alexander},
  journal={arXiv preprint arXiv:2602.19066},
  year={2026}
}
```

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
