# PSS-Net: Position–Scale–Survival State Dynamics for Infrared Small Target Detection

Official implementation of **PSS-Net** (IEEE Geoscience and Remote Sensing Letters, under review).

PSS-Net reformulates hierarchical feature learning for infrared small target detection (IRSTD)
as an explicit **target-state evolution** problem. At each encoder stage it estimates three
target states — position (P), scale (S), and survival confidence (Q) — and uses them to drive
adaptive feature transformation:

- **ADF** (Adaptive Deformable Field): scale-conditioned deformable sampling,
- **NSR** (Neural State Router): survival-aware, Q-guided resolution transition,
- **FDN** (Frequency Decomposition Neck): frequency-aware multi-scale fusion,
- **PCER** (Position-guided Context and Evidence Reconstruction): P-guided skip-feature
  selection in the decoder.

## Results (mIoU / Pd / Fa)

| Dataset   | mIoU  | Pd (%) | Fa (1e-6) | Params | FPS  |
|-----------|-------|--------|-----------|--------|------|
| NUAA-SIRST| 80.12 | 96.95  | 12.90     |        |      |
| NUDT-SIRST| 95.48 | 99.15  | 0.25      |        |      |
| IRSTD-1k  | 71.20 | 92.59  | 13.30     |        |      |

All configurations use 1.02\,M parameters.

## Installation

```bash
conda create -n pssnet python=3.10
conda activate pssnet
pip install torch torchvision  # PyTorch >= 2.0, CUDA >= 11.8
pip install -r requirements.txt
```

## Dataset preparation

Expected layout (splits follow the standard protocols):

```
dataset/
├── NUAA-SIRST/
│   ├── images/        # <name>.png
│   ├── masks/         # <name>.png
│   ├── trainval.txt   # one image name per line (no extension)
│   └── test.txt
├── NUDT-SIRST/  (same layout)
└── IRSTD-1k/    (same layout)
```

Set the dataset root with `--dataset-root` (default `./dataset`).

## Training

```bash
# NUAA-SIRST / NUDT-SIRST (256 x 256 inputs)
python train.py --dataset NUAA-SIRST --epochs 600 --test-start-epoch 200 \
    --model-config configs/nuaa_sirst.json --exp-name pssnet_nuaa

python train.py --dataset NUDT-SIRST --epochs 600 --test-start-epoch 200 \
    --model-config configs/nudt_sirst.json --exp-name pssnet_nudt

# IRSTD-1k (512 x 512 inputs)
python train.py --dataset IRSTD-1k --epochs 600 --test-start-epoch 200 \
    --base-size 512 --crop-size 512 \
    --model-config configs/irstd_1k.json --exp-name pssnet_irstd1k
```

The best model (`weight-<dataset>.pkl`), training log and checkpoints are saved under
`weight/<dataset>/GSFANet-<exp_name>-<timestamp>/` (the `GSFANet-` prefix is a legacy
artifact of the training script).

## Testing

```bash
python train.py --dataset NUDT-SIRST --mode test \
    --weight-path weight/NUDT-SIRST/<run_dir>/weight-NUDT-SIRST.pkl \
    --model-config configs/nudt_sirst.json
```

## Pretrained weights

Available in [Releases](../../releases) (one checkpoint per dataset).

## Citation

```bibtex
@article{mao2026pssnet,
  author  = {Mao, Enjie and Rao, Junmin and Zhu, Yibiao and Li, Fanming and Xu, Yiqing},
  title   = {{PSS-Net}: Position--Scale--Survival State Dynamics for Infrared Small Target Detection},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026},
  note    = {under review}
}
```

## License

This project is released under the MIT License.
