# BPT: Garment Mesh Reconstruction from Noisy Point Clouds

**MAC-3D-Garment Team** — Technical Report, June 2026

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](./main.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-Pending-red)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

---

## About This Repository

This is the collaborative workspace for the MAC-3D-Garment team. It contains:

- **Technical paper** — BPT fine-tuning for garment mesh reconstruction
- **Training pipeline** — fp32 stable protocol, lr warmup, NaN guard
- **Agent-orchestrated data processing** — A2A platform + Blender MCP + AutoResearch
- **Experiment logs** — V100 (4K tokens) baseline to H20 (10K tokens) results

---

## Abstract

Reconstructing clean quadrilateral garment meshes from noisy partial point clouds remains a critical challenge in 3D computer vision, with applications in virtual try-on, digital fashion design, and physics-based simulation. This technical report presents a fine-tuning pipeline based on Blocked and Patchified Tokenization (BPT), which encodes garment meshes into discrete token sequences and learns to reconstruct them from Michelangelo-encoded point cloud features. We describe an end-to-end agent-orchestrated data processing pipeline, a stable fp32 training protocol, and a systematic investigation of the sequence length bottleneck. Experiments on the ClothesNetM and Other_clothes datasets (2,807 training, 311 held-out test samples) demonstrate that while validation loss converges from 7.79 to 1.30 (83%), reconstruction quality is fundamentally limited by the maximum token sequence length during training. Our findings indicate that garment meshes require approximately 14,000 tokens for complete representation, exceeding the 10,000-token limit of our H20 96GB GPU.

---

## Key Results

| Metric | V100 (4K tokens) | H20 (10K tokens) | Required |
|--------|-----------------|-------------------|----------|
| Samples covered | 0% | ~15% | 95%+ |
| Validation loss | 1.49 | **1.30** | — |
| Token budget | 4,000 | 10,000 | **14,000** |
| Output | Fragments | Fragments | Complete meshes |

---

## Repository Structure

| Directory | Content |
|-----------|---------|
|  | LaTeX source + compiled PDF + figures |
|  | Training code (v6 final), model, config |
|  | A2A agent orchestration platform |
|  | QuadRemesher scripts, preprocessing pipeline |
|  | Weekly progress reports |
|  | Full AutoResearch workflow |
|  | Lessons learned & future plan |

## Citation



## Related

- [BPT](https://github.com/Tencent-Hunyuan/bpt) — Official implementation (CVPR 2025)
- [AutoResearch](https://github.com/karpathy/autoresearch) — karpathy's autonomous ML loop
