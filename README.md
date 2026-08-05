# SigLIP-Guided Residual Network for Medical Image Super-Resolution

This repository contains the implementation of a lightweight Super-Resolution (SR) framework designed to achieve semantic consistency in medical imaging. By leveraging Vision-Language Model (VLM) priors, the model addresses the critical trade-off between diagnostic fidelity and computational efficiency.

---

## Overview

Medical imaging modalities such as Endoscopy and Magnetic Resonance Imaging (MRI) are pivotal for clinical diagnosis but are frequently constrained by resolution limits due to hardware physics, scanning duration, and patient movement. High-resolution (HR) images are essential for identifying early-stage pathologies, such as mucosal lesions or fine anatomical brain structures.

Traditional super-resolution methods often utilize GANs, which risk the "hallucination" of false anatomical information. Alternatively, Transformer-based models can be computationally expensive and unsuitable for real-time applications. This project proposes a **SigLIP-Guided Residual Network**, a lightweight SR framework designed for semantic consistency.

### Key Features
* **Semantic Texture Optimization**: Minimizes the error between the semantic features of generated and target images using a Frozen SigLIP 2 Vision-Language model as a perceptual critic.
* **Hallucination Mitigation**: Prioritizes accurate super-resolution over simple "realism," reducing the risk of generating false anatomical instances.
* **Real-Time Ready**: Built with a compact footprint of approximately 0.96 million parameters (956,558 exactly), making it suitable for edge medical devices.
* **Cross-Modal Robustness**: Validated across multiple endoscopic datasets and MRI scans.

---

## Technical Methodology

### Architecture: ResidualSR
The network utilizes a Residual Difference Network architecture that learns to add high-frequency details to a bicubic base. The architecture comprises three stages:

1. **Shallow Feature Extraction**: Uses an initial convolutional layer to capture low-level image features.
2. **Deep Residual Mapping**: Consists of 8 Residual Blocks, each containing two convolutional layers with Batch Normalization and PReLU activation.
3. **High-Resolution Reconstruction**: Employs two sequential PixelShuffle (sub-pixel convolution) layers to upscale feature maps without the computational overhead of deconvolution.

### Optimization and Loss
* **Semantic Critic**: Employs a pre-trained, frozen SigLIP 2 model to guide training.
* **Feature Alignment**: Forces the model to learn authentic tissue textures by minimizing feature differences at mid-level network layers.
* **Hybrid Loss**: Combines semantic guidance with standard pixel-wise loss.
* **Optimization**: Uses the AdamW optimizer with a decaying learning rate over 50 epochs.

---

## Dataset Analysis

The following clinical datasets were utilized for training and validation:

| Modality | Dataset | Image Count | Purpose |
| :--- | :--- | :--- | :--- |
| **Endoscopic** | CVC-ClinicDB | 612 | In-Domain Training/Testing |
| **Endoscopic** | Kvasir-SEG | 1,000 | Zero-Shot Validation |
| **Endoscopic** | ETIS-Larib | 196 | Zero-Shot Validation |
| **MRI** | IXI T2 Dataset | ~16,000 | Structural Fidelity Testing |

**Preprocessing**: Images are split 90/10 for train-test sets. MRI files are converted from `.nii` format to `.png` slices.

---

## Experimental Results

### Endoscopic Super-Resolution (Quantitative)
| Dataset | Domain | Metric | Bicubic | Our Model |
| :--- | :--- | :--- | :--- | :--- |
| **CVC-ClinicDB** | In-Domain | PSNR (dB) | 28.62 | 33.92 |
| | | SSIM | 0.8592 | 0.9094 |
| | | LPIPS | 0.2549 | 0.1619 |
| **Kvasir-SEG** | Zero-Shot | PSNR (dB) | 27.45 | 28.49 |
| | | SSIM | 0.7990 | 0.8255 |
| **ETIS-Larib** | Zero-Shot | PSNR (dB) | 30.96 | 31.99 |
| | | SSIM | 0.8714 | 0.8801 |

### MRI Super-Resolution (Quantitative)
| Metric | Bicubic | Our Model |
| :--- | :--- | :--- |
| **PSNR (dB)** | 23.10 | 25.69 |
| **SSIM** | 0.4530 | 0.8394 |
| **LPIPS** | 0.4166 | 0.1526 |

The model demonstrates a 5.30 dB gain in signal fidelity for endoscopy and a significant boost in structural similarity for MRI. Qualitative results confirm the recovery of complex mucosal textures and fine brain structural details.

---

## References
1. Huang, W. et al. "Versatile and efficient medical image super-resolution via frequency-gated mamba." BIBM, 2025.
2. Ledig, C. et al. "Photo-realistic single image super-resolution using a generative adversarial network." CVPR, 2016.
3. Liang, J. et al. "Swinir: Image restoration using swin transformer." ICCVW, 2021.
4. Luo, Y. et al. "Probabilistic prior-guided anatomical alignment for mri super-resolution." MICCAI, 2025.
5. Tschannen, M. et al. "Siglip 2: Multilingual vision-language encoders with improved semantic understanding..." ArXiv, 2025.
