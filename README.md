# DL Research Mini-Project

This repository contains a single Jupyter notebook with two independent deep learning experiments: solving a 3D Partial Differential Equation using a physics-informed neural network architecture (SPINN), and classifying urban environmental sounds using CNNs trained on mel-spectrograms. The notebook was developed and executed on Kaggle.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Part 1 — SPINN: Solving the 3D Helmholtz Equation](#part-1--spinn-solving-the-3d-helmholtz-equation)
  - [Overview](#overview)
  - [Theoretical Background](#theoretical-background)
  - [PDE Formulation](#pde-formulation)
  - [Architecture](#spinn-architecture)
  - [Loss Function](#loss-function)
  - [Training Pipeline](#training-pipeline)
  - [Evaluation](#evaluation)
  - [Key Findings](#key-findings)
- [Part 2 — UrbanSound8K: Audio Classification](#part-2--urbansound8k-audio-classification)
  - [Overview](#overview-1)
  - [Dataset](#dataset)
  - [Audio Preprocessing Pipeline](#audio-preprocessing-pipeline)
  - [Model 1: Custom CNN Baseline](#model-1-custom-cnn-baseline)
  - [Model 2: ResNet18 with Transfer Learning](#model-2-resnet18-with-transfer-learning)
  - [Training Setup](#training-setup)
  - [Evaluation](#evaluation-1)
  - [Key Findings](#key-findings-1)
- [Dependencies](#dependencies)
- [References](#references)

---

## Repository Structure

```
dl-research-mini-project/
│
├── SPINN Notebook.ipynb    # Main notebook — both experimental parts
│
├── spinn_adam.pth          # SPINN weights after Adam optimisation phase
├── spinn_LBFGS.pth         # SPINN weights after L-BFGS fine-tuning phase
│
├── best_model.pth          # Best checkpoint of the custom UrbanSoundCNN
└── cnn.pth                 # ResNet18 checkpoint saved when val_acc > 90%
```

| File | Description |
|---|---|
| `SPINN Notebook.ipynb` | Complete research notebook. Part 1 covers physics-informed learning for the 3D Helmholtz equation. Part 2 covers supervised audio classification on UrbanSound8K using two CNN architectures. |
| `spinn_adam.pth` | Saved model state after the Adam warm-up training stage (1 500 epochs, lr=3e-4, λ_bc=10.0). Used as the starting point for L-BFGS fine-tuning. |
| `spinn_LBFGS.pth` | Final SPINN weights produced by three stages of L-BFGS optimisation (max_iter=300 per stage, strong Wolfe line search, λ_bc=50.0, NC=40³). |
| `best_model.pth` | Best-performing `UrbanSoundCNN` checkpoint as measured by validation accuracy. Saved every time a new val_acc record is set. |
| `cnn.pth` | ResNet18-based model checkpoint. Saved conditionally when val_acc exceeds 90%, indicating a high-quality model. |

---
---

# Part 1 — SPINN: Solving the 3D Helmholtz Equation

## Overview

The first part of the notebook investigates **Physics-Informed Neural Networks (PINNs)** and the **Separable PINN (SPINN)** architecture as a mesh-free method for solving partial differential equations. The target problem is the three-dimensional **Helmholtz equation** on the unit cube `[-1, 1]³`, chosen because it admits a closed-form analytical solution. This enables rigorous, quantitative error measurement throughout training without relying on any external numerical solver.

The notebook implements the SPINN architecture from scratch in PyTorch, defines the physics-informed composite loss, and trains the model in two stages — first with Adam, then with L-BFGS — following the recommended procedure from the original paper.

---

## Theoretical Background

### Physics-Informed Neural Networks (PINNs)

A standard neural network is trained to minimise the discrepancy between its predictions and a set of labelled observations. A **PINN** extends this by also penalising violations of a governing physical law (a PDE) at a set of interior points called **collocation points**. The total training loss is:

```
L_total = L_data + λ · L_PDE
```

where `L_data` handles boundary and initial conditions and `L_PDE` is the **PDE residual** evaluated via automatic differentiation. Because the network is constrained to satisfy the physics, it can generalise well even with very few or no labelled solution values — the physics itself serves as the primary source of supervision.

### Why SPINN?

The key bottleneck of classical PINNs in multiple spatial dimensions is the **curse of dimensionality**: for a `d`-dimensional domain, covering it uniformly with `N` points per axis requires `N^d` collocation points in total. For `d=3` and `N=32`, that means 32 768 points — already expensive — and it scales catastrophically for higher `d` or finer resolution.

**SPINN** (Separable PINN, Cho et al. 2023) solves this by exploiting a **separable structure** in the solution representation:

- Instead of one large network mapping `(x, y, z) → u`, three small **coordinate-wise sub-networks** each map a single scalar `xᵢ → ℝʳ`.
- Their outputs are combined via a **CP (Canonical Polyadic) tensor decomposition**, reducing the total computational cost from `O(N^d)` to `O(N · d)`.
- Collocation points are arranged on a **regular lattice** (meshgrid), which fits naturally into the separable structure and enables efficient PDE residual computation using forward-mode automatic differentiation.

The result is a model that can train with 64 000 collocation points on a single consumer GPU in minutes rather than hours.

---

## PDE Formulation

The target equation is the **3D Helmholtz equation**:

```
∇²u(x,y,z) + k²u(x,y,z) = q(x,y,z),    (x,y,z) ∈ [-1,1]³
```

where:
- `u(x,y,z)` — the unknown scalar field (e.g., wave amplitude or acoustic pressure)
- `k` — the wave number (set to `k=1.0` in the notebook)
- `q(x,y,z)` — the source term (right-hand side), derived analytically from the manufactured solution

**Boundary conditions** — homogeneous Dirichlet on all six faces of the cube:

```
u = 0   on ∂Ω  (all six faces: x=±1, y=±1, z=±1)
```

This corresponds physically to a membrane fixed at all edges, or a standing wave pattern confined to the domain.

### Manufactured / Analytical Solution

To enable exact error measurement, the notebook uses the **method of manufactured solutions**. The analytical reference is:

```
u*(x,y,z) = sin(a₁πx) · sin(a₂πy) · sin(a₃πz)
```

with `a₁ = a₂ = a₃ = 1`. Substituting into the Helmholtz equation gives the corresponding source term analytically:

```
q(x,y,z) = [-(a₁π)² - (a₂π)² - (a₃π)² + k²] · u*(x,y,z)
```

Both functions are implemented inside the `BaseModel` class as `reference_solution()` and `function_of_source()`. Note that `u*` satisfies the zero Dirichlet boundary condition exactly, since `sin(nπ·(±1)) = 0`.

---

## SPINN Architecture

The model is built in two layers of abstraction:

### `BaseModel` — Problem Container

This class stores all problem-specific information and handles the physics:
- Domain bounds `x, y, z ∈ [-1, 1]` and PDE parameters `k`, `a₁`, `a₂`, `a₃`
- **Pre-computed boundary points** for all six faces, registered as PyTorch buffers (`register_buffer`) so they are automatically moved to the correct device. Each face contributes `N×N = 32×32 = 1 024` points, totalling `6 144` boundary points.
- `calculate_pde_loss()`: computes second-order partial derivatives `∂²u/∂x²`, `∂²u/∂y²`, `∂²u/∂z²` via two applications of `torch.autograd.grad`, assembles the full Helmholtz residual `∇²u + k²u - q`, and returns its MSE.
- `calculate_bc_loss()`: evaluates the model on all six boundary faces and returns MSE against zero.

### `BaseSPINN` — Separable Architecture

```
Three independent body networks, one per spatial coordinate:

    body_x:  ℝ → ℝʳ      (processes only x)
    body_y:  ℝ → ℝʳ      (processes only y)
    body_z:  ℝ → ℝʳ      (processes only z)

Each body network is a Tanh-MLP:
    Linear(1 → 64) → Tanh
    Linear(64 → 64) → Tanh   (repeated n_hidden_layers - 1 times)
    Linear(64 → r)             (r = rank = 4, no activation)

Forward pass — CP decomposition:
    fx = body_x(x)                    # shape: (N, r)
    fy = body_y(y)                    # shape: (N, r)
    fz = body_z(z)                    # shape: (N, r)
    û  = (fx * fy * fz).sum(dim=-1)   # element-wise product, sum over rank
```

All weights are initialised with **Xavier uniform** scaled for the Tanh gain (`nn.init.calculate_gain("tanh")`), which prevents gradient vanishing and explosion in deep Tanh networks.

**Configured hyperparameters:**

| Parameter | Value |
|---|---|
| Rank `r` | 4 |
| Hidden dimension | 64 |
| Number of hidden layers | 4 |
| Activation function | Tanh |
| Weight initialisation | Xavier uniform (Tanh gain) |
| Wave number `k` | 1.0 |
| Frequency parameters | a₁=a₂=a₃=1 |
| Boundary grid per face | 32×32 = 1 024 points |
| Total boundary points | 6 × 1 024 = 6 144 |

---

## Loss Function

The total training objective combines two terms:

```
L_total = L_PDE + λ_bc · L_BC
```

**PDE residual loss** — mean squared error of the Helmholtz residual at all collocation points:

```
L_PDE = (1/Nc) Σᵢ |∇²û(xᵢ) + k²û(xᵢ) - q(xᵢ)|²
```

The Laplacian `∇²û` is computed exactly via two rounds of `torch.autograd.grad`. This requires no finite differences and produces machine-precision gradients at arbitrary points in the domain.

**Boundary condition loss** — MSE of the predicted values on all six boundary faces:

```
L_BC = (1/Nb) Σⱼ |û(x_boundary_j)|²
```

The **boundary penalty weight `λ_bc`** is a critical hyperparameter controlling the trade-off between PDE satisfaction and boundary enforcement:

| Training Stage | `λ_bc` | Rationale |
|---|---|---|
| Adam warm-up, 1 000 epochs | 0.0 | Model freely explores the PDE solution space, no boundary competition |
| Adam main run, 1 500 epochs | 10.0 | Boundary conditions introduced gradually to shape the solution |
| L-BFGS fine-tuning, 3 stages | 50.0 | Strong boundary enforcement for precise satisfaction at the final stage |

Starting with `λ_bc=0` implements a **curriculum**: the model first learns the bulk physics, then boundary constraints are progressively enforced. This avoids early saddle-point dynamics where boundary and PDE objectives conflict and the optimiser stagnates.

---

## Training Pipeline

Training follows a **two-stage curriculum**.

### Collocation Point Resampling

Every `resample_iter` iterations, a fresh lattice of interior collocation points is generated:

```python
N_per_axis = int(NC ** (1/3))        # e.g. 32 for NC=32³
x, y, z = torch.linspace(-1, 1, N_per_axis)  # one axis each
xx, yy, zz = torch.meshgrid(x, y, z)         # full 3D grid
```

All three coordinate tensors have `requires_grad=True` to enable autograd through the PDE residual. Resampling prevents the model from memorising specific lattice points and improves domain coverage.

### Stage 1 — Adam Warm-up

```
Run 1:   epochs=1 000,  lr=1e-3,   λ_bc=0.0,   NC=32³=32 768,  resample_iter=100
Run 2:   epochs=1 500,  lr=3e-4,   λ_bc=10.0,  NC=32³=32 768,  resample_iter=50
```

Adam handles noisy gradients well and converges quickly in early training. The first run with `λ_bc=0` allows unconstrained exploration of the PDE solution space. The second run at a lower learning rate with boundary enforcement begins shaping the solution toward the correct zero-boundary behaviour.

### Stage 2 — L-BFGS Fine-tuning

```
Stages:         3  (with collocation resampling between each stage)
max_iter:       300 per stage
history_size:   50  (past gradients stored for Hessian approximation)
line_search:    Strong Wolfe conditions
λ_bc:           50.0
NC:             40³ = 64 000
```

L-BFGS is a quasi-Newton method that approximates the inverse Hessian from a sliding window of past gradient differences. It converges much faster than Adam near a minimum but requires a smooth loss landscape — satisfied here because Adam has already brought the model into a good basin. The strong Wolfe line search guarantees both sufficient decrease and curvature conditions on every step.

The three-stage structure with intermediate resampling prevents L-BFGS from over-specialising to a single collocation grid while making steady progress in reducing the PDE and boundary residuals.

### Validation During Training

Every `log_iter` steps, the model is evaluated in `torch.no_grad()` mode on a fixed test grid of `50×50×50 = 125 000` points. The **relative L2 error** is computed and logged:

```
error = ||û - u*||₂ / ||u*||₂
```

Loss curves (total, PDE residual, boundary) and the error curve are displayed on logarithmic axes. The 3D solution field is visualised as a scatter plot coloured by the `seismic` colormap, allowing qualitative comparison against the analytical reference plotted at the start.

---

## Evaluation

Final evaluation after L-BFGS computes the MSE between the model's prediction and the analytical solution on the `50×50×50` test grid:

```python
best_u_3d = best_u.reshape(u_test.shape)
mse = ((u_test.cpu().numpy() - best_u_3d.cpu().numpy()) ** 2).mean()
print(mse)
```

---

## Key Findings

- **SPINN successfully recovers the 3D Helmholtz solution** without any labelled interior data, driven entirely by the PDE residual and boundary conditions.

- **The two-stage Adam → L-BFGS strategy is essential**: Adam provides stable warm-up that prevents L-BFGS from stalling in early saddle points; L-BFGS achieves the precision that first-order methods cannot reach at small loss values.

- **`λ_bc` scheduling is critical**: beginning at `λ_bc=0` allows free PDE-space exploration, and the gradual increase (0 → 10 → 50) prevents premature boundary enforcement from creating conflicting gradients that stall training.

- **Collocation resampling every 50–100 iterations** is an important regularisation technique that prevents overfitting to specific lattice points and improves spatial coverage.

- **The separable CP decomposition** makes 3D PDE solving tractable on a single GPU. The `O(N·d)` collocation cost compared to `O(N^d)` for classical PINNs is a fundamental architectural advantage that grows more pronounced with dimension.

- **The manufactured solution approach** provides a clean, fully reproducible benchmark: constructing the source term analytically from a known `u*` allows continuous monitoring of the true approximation error throughout training.

---
---

# Part 2 — UrbanSound8K: Audio Classification

## Overview

The second part switches from physics-informed learning to a **standard supervised classification** task. The goal is to identify which of 10 urban sound categories is present in a short audio clip, using the well-known **UrbanSound8K** benchmark dataset. Audio signals are transformed into mel-spectrograms and treated as single-channel images fed into convolutional neural networks.

Two architectures are explored: a **custom CNN trained from scratch** as a baseline, and a **ResNet18 with ImageNet pre-trained weights** adapted for single-channel spectrogram input. The full pipeline covers data loading, on-the-fly augmentation, model training with early stopping and learning rate scheduling, and detailed per-class evaluation.

---

## Dataset

**UrbanSound8K** is a standard benchmark for environmental audio classification.

| Property | Value |
|---|---|
| Total clips | 8 732 |
| Number of classes | 10 |
| Maximum clip duration | 4 seconds |
| Sampling rate used | 22 050 Hz |
| Pre-defined folds | 10 |
| Source | [urbansounddataset.weebly.com](https://urbansounddataset.weebly.com/urbansound8k.html) |

**Classes:**

| Label | Description |
|---|---|
| `air_conditioner` | HVAC system hum |
| `car_horn` | Vehicle horn beep |
| `children_playing` | Kids playing outdoors |
| `dog_bark` | Dog vocalisation |
| `drilling` | Power drill operation |
| `engine_idling` | Idling motor |
| `gun_shot` | Firearm discharge |
| `jackhammer` | Pneumatic hammer |
| `siren` | Emergency vehicle siren |
| `street_music` | Street musician performance |

The dataset ships with 10 pre-defined folds designed for cross-validation. In this notebook, **fold 10 is reserved exclusively as the test set**, and the remaining 9 folds are further split into train and validation subsets in a stratified manner. This strict separation prevents any data leakage that would arise from mixing fold boundaries.

**Expected on-disk structure:**

```
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   └── ... fold10/
└── metadata/
    └── UrbanSound8K.csv
```

---

## Audio Preprocessing Pipeline

All audio processing is encapsulated in the `UrbanSoundDataset` class, which inherits from `torch.utils.data.Dataset`.

### Step 1 — Loading and Length Normalisation

Each audio file is loaded with `librosa.load(path, sr=22050)`, resampling if necessary. Clips vary in duration, so all waveforms are normalised to exactly `4 × 22 050 = 88 200` samples:

- **Too short** → zero-padding appended at the end (`np.pad(..., mode="constant")`)
- **Too long** → hard truncation to 88 200 samples

This guarantees a uniform temporal dimension for batch processing.

### Step 2 — Mel-Spectrogram Extraction

Each waveform is converted to a mel-spectrogram:

```python
librosa.feature.melspectrogram(y=y, sr=22050, n_fft=2048, hop_length=512, n_mels=128, fmax=11025)
```

| Parameter | Value | Description |
|---|---|---|
| `n_fft` | 2048 | FFT window size (~93 ms at 22 kHz) |
| `hop_length` | 512 | Frame step (~23 ms), controls time resolution |
| `n_mels` | 128 | Number of mel filterbank channels |
| `fmax` | 11 025 Hz | Maximum frequency (Nyquist) |

The power spectrogram is converted to decibels: `librosa.power_to_db(S, ref=np.max)`, producing values in approximately `[-80, 0]` dB. This logarithmic compression aligns with human auditory perception and compresses the dynamic range, making discriminative patterns more accessible to the network. Output shape: `(128, T)` where `T ≈ 173` frames.

### Step 3 — Z-score Normalisation

Each spectrogram is individually normalised per sample:

```python
mel_norm = (mel - mel.mean()) / (mel.std() + 1e-6)
```

This per-sample standardisation removes the effect of absolute loudness and recording conditions, allowing the network to focus on spectral shape and temporal patterns rather than overall volume.

### Step 4 — Output Tensor

The normalised spectrogram is converted to a float tensor with an added channel dimension: shape `(1, 128, 173)`. This single-channel "image" is directly compatible with Conv2d layers. The class label is encoded with `sklearn.LabelEncoder` and returned as a `torch.long` scalar.

### On-the-fly Augmentations (Train Split Only)

| Augmentation | Probability | Description |
|---|---|---|
| **Time shift** | 0.5 | Circular roll of the waveform by a random integer in `[-sr, sr]` samples, shifting the temporal onset position |
| **Additive Gaussian noise** | 0.5 | White noise scaled to `0.005 × Uniform(0,1) × max(|y|)`, simulating background noise variability |

Augmentations are applied on-the-fly and disabled for validation and test loaders.

### Data Splits

```
Fold 10                               →  Test   (~  873 samples, ~10%)
Folds 1–9, stratified, 85% portion   →  Train  (~6 663 samples, ~76%)
Folds 1–9, stratified, 15% portion   →  Val    (~1 175 samples, ~14%)
```

`DataLoader` settings: batch_size=1 024, num_workers=4, pin_memory=True. Validation and test loaders have `shuffle=False` for reproducible metric computation.

---

## Model 1: Custom CNN Baseline

A lightweight convolutional network trained from scratch to establish a performance floor.

### Architecture

```
Input: (B, 1, 128, 173)
  │
  ├── Conv2d(1 →  32, 3×3, pad=1) → BatchNorm2d(32)  → ReLU → MaxPool2d(2×2)
  ├── Conv2d(32 → 64, 3×3, pad=1) → BatchNorm2d(64)  → ReLU → MaxPool2d(2×2)
  ├── Conv2d(64 →128, 3×3, pad=1) → BatchNorm2d(128) → ReLU → MaxPool2d(2×2)
  └── Conv2d(128→256, 3×3, pad=1) → BatchNorm2d(256) → ReLU → MaxPool2d(2×2)
  │
  ├── AdaptiveAvgPool2d(1×1)   ← collapses spatial dims to a single vector
  │
  └── Flatten → Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→10)

Output: class logits (B, 10)
```

Four convolutional blocks progressively double channel depth while halving spatial resolution via `MaxPool2d`. `AdaptiveAvgPool2d(1,1)` makes the model resolution-agnostic, useful if input length varies. `BatchNorm2d` after each convolution stabilises activations and provides implicit regularisation. The classifier head uses `Dropout(p=0.5)` to prevent co-adaptation of neurons.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimiser | Adam, lr=1e-3, weight_decay=1e-4 |
| Loss function | CrossEntropyLoss |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5 epochs) |
| Early stopping | patience=15 epochs without val_acc improvement |
| Max epochs | 50 |
| Checkpoint rule | Save `best_model.pth` on every new val_acc record; save `cnn.pth` when val_acc > 70% |

---

## Model 2: ResNet18 with Transfer Learning

A significantly stronger model that reuses low-level feature representations learned on ImageNet. Model states are saved here https://drive.google.com/drive/folders/1lSveNPOij0RziTEEHhg01OV_k0ktIyGF?usp=share_link

### Motivation for Transfer Learning

ResNet18 pre-trained on ImageNet has learned a rich hierarchy of low-level visual features — oriented edges, frequency gratings, texture detectors, and blob detectors. These are not specific to natural photographs: mel-spectrograms exhibit structurally similar local patterns (harmonic lines, onset transients, textured frequency bands). Transfer learning allows the network to skip expensive low-level feature learning and focus optimisation on task-specific high-level representations, which is especially valuable when the target dataset is small (~6 600 training samples).

### Architecture Adaptation

Standard ResNet18 expects 3-channel (RGB) images, but mel-spectrograms are single-channel. The adaptation strategy:

```
1. Load ResNet18 with pretrained=True (ImageNet weights)

2. Replace conv1:
   Original:  Conv2d(3, 64, kernel=7×7, stride=2, padding=3, bias=False)
   New:       Conv2d(1, 64, kernel=7×7, stride=2, padding=3, bias=False)

3. Initialise new conv1 weights from the pretrained weights by channel averaging:
      new_weight = pretrained_weight.mean(dim=1, keepdim=True)
   This preserves the spatial structure and magnitude of pretrained filters.

4. Feature extractor = all ResNet18 layers except the final FC:
   conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool
   Output: (B, 512, 1, 1)

5. Custom classifier head:
   Flatten → Linear(512 → 256) → ReLU → Dropout(0.5) → Linear(256 → 10)
```

Channel-averaging the pretrained RGB weights is the principled way to convert a 3-channel pretrained model to grayscale: it preserves the effective receptive field behaviour and feature magnitudes while requiring only a single forward-compatible change.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimiser | AdamW, lr=3e-4, weight_decay=1e-4 |
| Loss function | CrossEntropyLoss with label_smoothing=0.1 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=4 epochs) |
| Early stopping | patience=15 epochs without val_acc improvement |
| Max epochs | 50 |
| Backbone frozen | No — full end-to-end fine-tuning |
| Checkpoint rule | Save `best_model.pth` on every new record; save `cnn.pth` when val_acc > 90% |

**Label smoothing** (`label_smoothing=0.1`) replaces hard one-hot targets with soft targets `(1-ε)·y_hard + ε/K`, preventing overconfident predictions and improving calibration on the test set. **AdamW** decouples the weight decay from the adaptive gradient scaling (fixing a known issue in vanilla Adam), which produces better-regularised weights during fine-tuning.

---

## Training Setup

Both models share the same training loop structure.

### `train_epoch`

For each mini-batch: forward pass → cross-entropy loss → zero gradients → backward → optimiser step. Accumulates running loss and correct prediction counts. Returns epoch-mean loss and accuracy.

### `validate`

Same structure wrapped in `torch.no_grad()` and `model.eval()` (disables dropout and batch-norm running-stat updates). Returns val loss and accuracy.

### Learning Rate Scheduling

`ReduceLROnPlateau` monitors validation loss after each epoch. When the loss fails to decrease for `patience` consecutive epochs, the learning rate is multiplied by `factor=0.5`. This automatic step-size adaptation avoids manual tuning of learning rate schedules and is well-suited to training with early stopping.

### Early Stopping

A `patience_counter` tracks consecutive epochs without a new validation accuracy record. When it reaches the patience threshold (15 epochs for both models), training terminates and the best checkpoint is used for evaluation. This prevents overfitting and saves computation.

---

## Evaluation

After training, the best checkpoint is reloaded (`model.load_state_dict(torch.load('best_model.pth'))`) and the model is evaluated in `model.eval()` mode on the entire held-out test fold.

### Metrics and Visualisations

**Test Accuracy** — global accuracy over all test samples, printed as a percentage.

**Classification Report** (`sklearn.metrics.classification_report`, 4 decimal places) — per-class precision, recall, and F1-score. Reveals systematic under-performance on specific classes.

**Confusion Matrix** — `10×10` heatmap (seaborn `Blues`) with raw prediction counts. Off-diagonal cells indicate misclassifications. Rendered with rotated x-axis labels and a colour-bar.

**Per-Class Accuracy Bar Chart** — individual bar per class with percentage accuracy annotated above each bar. A horizontal dashed red line marks the overall test accuracy for reference.

---

## Key Findings

- **Audio-as-image via mel-spectrograms** is a simple but highly effective strategy. The perceptually-motivated mel scale compresses high-frequency content (where less discriminative information resides) and expands low-frequency detail, making spectral patterns more network-friendly than a linear-frequency spectrogram.

- **Per-sample z-score normalisation** is important for robustness: it removes the effect of absolute recording volume and ensures that all inputs to the network have comparable activation magnitudes, independent of microphone placement or source distance.

- **Transfer learning from ImageNet to mel-spectrograms** provides a substantial accuracy boost over training from scratch, despite the domain mismatch. The low-level feature detectors in ResNet18 — originally tuned for natural image textures and edges — transfer well to the local spectro-temporal structure in mel-spectrograms.

- **Label smoothing** reduces overconfidence on training examples and improves generalisation. It is particularly beneficial for the ResNet18 model, which has far more capacity and is at greater risk of memorising training labels.

- **Fold-based test separation** (fold 10 entirely held out) ensures an unbiased estimate of generalisation. The stratified train/val split prevents class imbalance from distorting early-stopping decisions.

- **On-the-fly time-shift and noise augmentations** improve robustness to recording variability, onset timing, and background conditions, without inflating storage by pre-generating an augmented dataset.

- Acoustically similar class pairs — particularly `engine_idling` vs `air_conditioner` (both produce steady low-frequency hum) and `jackhammer` vs `drilling` (both produce impulsive rhythmic noise) — are the hardest to separate. Their mel-spectrograms share similar stationary texture patterns, and most residual misclassification errors are concentrated within these pairs, as visible in the confusion matrix.

---

## Dependencies

```bash
pip install torch torchvision torchaudio
pip install librosa
pip install scikit-learn
pip install pandas numpy matplotlib seaborn
pip install tqdm
```

The notebook was executed on Kaggle with GPU acceleration (CUDA). All model checkpoints were saved to `/kaggle/working/` and are committed to the repository root.

Used AI for code refactoring and formalization of the text.
