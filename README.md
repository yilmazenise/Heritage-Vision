# Heritage-Vision

Semantic segmentation of historical architectural floor plans using U-Net family
architectures, transfer learning and knowledge distillation.

Heritage-Vision takes a scanned or digitised plan drawing of a historical building
and labels every pixel as **wall**, **iwan**, **room** or **background**. The goal is
to turn heritage plan archives — which exist mostly as raster drawings — into
machine-readable spatial data that can support architectural analysis, typological
comparison and digital documentation of cultural heritage.

The central problem the project addresses is **data scarcity**: annotated historical
plans are expensive to produce and exist only in small numbers (131 annotated plans
here). Rather than training from scratch, every architecture in this repository is
bootstrapped from a large pre-trained segmentation network through two competing
transfer strategies, which are then compared head to head.

---

## Table of Contents

- [Problem and Approach](#problem-and-approach)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Transfer Strategies](#transfer-strategies)
- [Training Setup](#training-setup)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Authors and Mentorship](#authors-and-mentorship)
- [License](#license)

---

## Problem and Approach

Architectural plans are not natural images. They are sparse, high-contrast line
drawings where thin strokes carry most of the semantic weight, and where the classes
of interest (rooms, iwans) are large homogeneous regions delimited by those thin
strokes. A segmentation model must therefore preserve fine spatial detail while
reasoning about large-scale layout — exactly the trade-off the U-Net encoder/decoder
with skip connections is designed for.

The pipeline is:

1. **Annotation** — plans are labelled in LabelMe and exported as JSON.
2. **Rasterisation** — JSON polygons are converted into paired `*Plan.jpg` /
   `*Seg.jpg` images.
3. **Class mapping** — greyscale mask intensities are collapsed into four integer
   class labels.
4. **Pre-training** — a `convnext_base`-backed U-Net is pre-trained as the knowledge
   source.
5. **Transfer** — each target architecture is trained twice: once with copied
   weights, once by distilling from the pre-trained teacher.
6. **Evaluation** — mIoU and pixel accuracy on a held-out test split, with side-by-side
   visualisations.

## Dataset

| Property | Value |
|---|---|
| Total annotated plans | 131 |
| Train / test split | 104 / 27 (80 % / 20 %) |
| Input | RGB plan image, `*Plan.jpg` |
| Target | Greyscale mask, `*Seg.jpg` |
| Working resolution | 512 × 512 (train/test transform) |
| Classes | 4 |

The dataset itself is **not included** in this repository — only the code that
produces and consumes it.

### Class definitions

| Label | Class | Visualisation colour |
|---|---|---|
| `0` | Walls | `(0, 0, 0)` black |
| `1` | Iwan | `(255, 80, 80)` red |
| `2` | Room | `(80, 80, 255)` blue |
| `3` | Background | `(255, 255, 255)` white |

The *iwan* — a vaulted hall walled on three sides and open to a courtyard on the
fourth — is treated as a class of its own rather than folded into "room", because it
is the organising element of the plan types this dataset covers.

### Annotation pipeline

`dataset/labelme.ipynb` reads LabelMe JSON exports, decodes the embedded base64
images, and rasterises the polygon annotations into colour masks using an intermediate
class palette (`banyo`, `room`, `eyvan`). `dataset/split_test_train.ipynb` then pairs
each plan with its mask and performs a stratified 80/20 split into `images/train` and
`images/test`.

At load time, `dataset/dataset.py` discovers samples by filename convention — every
file ending in `Plan.jpg` is an input, and its target is the same name with
`Plan.jpg` replaced by `Seg.jpg`. Mask intensities are then binned into the four
class indices:

```python
mask[(mask_original <= 25.0)]                             = 0.0  # Walls
mask[(mask_original >= 26.0)  & (mask_original <= 120.0)] = 1.0  # Iwan
mask[(mask_original >= 121.0) & (mask_original <= 230.0)] = 2.0  # Room
mask[(mask_original >= 231.0)]                            = 3.0  # Background
```

Note that the networks emit a **single continuous channel** (`out_channels=1`) that is
regressed towards these class indices and rounded at inference, rather than the more
usual 4-channel softmax. This keeps the output head small, which matters at this
dataset size.

### Augmentation

Training uses Albumentations: horizontal flip (p = 0.5), vertical flip (p = 0.5) and
rotation (p = 0.5), followed by ImageNet normalisation and a resize to 512 × 512.
The test transform applies normalisation and resize only. Flips and rotations are
safe here because a floor plan has no canonical orientation.

## Model Architectures

All four architectures share the U-Net skeleton — a contracting encoder, an expanding
decoder built from `ConvTranspose2d` upsampling, skip connections concatenated across
the bottleneck, and a 1 × 1 output convolution — but differ in how the encoder blocks
are built.

| Module | Class | Encoder block design |
|---|---|---|
| `models/unet.py` | `UNET` | Baseline double 3 × 3 conv + BatchNorm + ReLU |
| `models/resunet18.py` | `RESUNET18` | Residual double conv with a 1 × 1 projection shortcut, ResNet-18 style |
| `models/resunet50.py` | `RESUNET50` | Triple-conv bottleneck blocks, ResNet-50 style |
| `models/vgg16.py` | `VGGUNET16` | VGG-16 style stages — deeper stacks of 3 × 3 convs per resolution level |
| `models/vgg19.py` | `VGGUNET19` | VGG-19 style stages — deepest convolution stacks |

Every model takes the same constructor signature:

```python
model = VGGUNET16(features=64, in_channels=3, out_channels=1)
```

The decoder pads asymmetric feature maps before concatenation, so non-power-of-two
input sizes are handled without cropping the skip connection.

`RESUNET50` is implemented and available but has no accompanying training notebook in
this repository.

## Transfer Strategies

The pre-trained source is a U-Net with a `convnext_base` backbone
(`pretrained-backbones-unet`), pre-trained by `pretraining/pretraining.py` and
`pretraining/pretrain.ipynb` at a learning rate of 1e-3 and stored under
`checkpoints/pretrained/`.

Each architecture is then trained twice, from the same starting point, under two
strategies:

**Model-1 — Weight Copy.** Parameters whose names match between the pre-trained
network and the target architecture are copied directly; the rest are randomly
initialised. Training then proceeds on the segmentation loss alone.

```python
for k, v in pretrained_dict.items():
    if k in model1_dict:
        new_state_dict1[k] = v
```

**Model-2 — Knowledge Distillation.** The pre-trained network is frozen and used as a
teacher. The student is trained on the sum of the segmentation loss against ground
truth and a distillation loss against the teacher's predictions, both computed as
per-pixel MSE:

```python
criterion = nn.MSELoss(reduction='none')
# segmentation loss:  criterion(pred_student, mask).mean()
# distillation loss:  criterion(pred_student, teacher_model(img_input)).mean()
```

Running both for every architecture is what makes the results table below a controlled
comparison: same data, same schedule, same seed — only the transfer mechanism differs.

## Training Setup

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam, β = (0.9, 0.999) |
| Learning rate | 1e-4 (1e-3 for pre-training) |
| Weight decay | 1e-4 |
| Scheduler | `ReduceLROnPlateau`, mode `min`, patience 20, factor 0.1 |
| Loss | Per-pixel MSE (+ distillation term for Model-2) |
| Epochs | 150 |
| Batch size | 8 |
| Gradient clipping | 1.0 |
| Seed | 9999 |
| Input size | 512 × 512 |

`seed_everything()` fixes Python, NumPy and CUDA seeds and sets
`cudnn.deterministic = True`, so runs are reproducible.

Checkpoints are written by `utils/save_load_utils.py` and carry the full
hyperparameter dictionary, optimiser state, best-model flag, last epoch and timestamp
alongside the weights — a checkpoint is self-describing and can be audited without the
notebook that produced it.

### Metrics

- **mIoU** — intersection over union computed per class and averaged over the four
  classes (`utils/evaluation_utils.py`). Where a class is absent from both prediction
  and target, IoU is defined as 1.0.
- **Pixel accuracy** — share of correctly labelled pixels after rounding.

## Results

Best checkpoint per run, evaluated on the 27-plan test split.

| Architecture | Strategy | Best epoch | Eval loss | mIoU (%) | Pixel acc. (%) |
|---|---|---|---|---|---|
| U-Net | Weight Copy | 141 | 0.0573 | 89.73 | 97.23 |
| U-Net | Knowledge Distillation | 129 | 0.0692 | 88.45 | 96.41 |
| ResUNet-18 | Weight Copy | 146 | 0.0612 | 89.25 | 97.14 |
| ResUNet-18 | Knowledge Distillation | 140 | 0.0645 | 89.62 | 96.90 |
| VGGUNet-16 | Weight Copy | 133 | 0.0255 | 94.45 | 98.74 |
| VGGUNet-16 | Knowledge Distillation | 135 | 0.0315 | 94.11 | 98.95 |
| VGGUNet-19 | Weight Copy | 110 | 0.0235 | 94.75 | 98.86 |
| **VGGUNet-19** | **Knowledge Distillation** | **100** | **0.0253** | **95.23** | **99.18** |

Reading the table:

- **The VGG-style encoders win decisively.** Both VGGUNet variants clear 94 % mIoU
  while U-Net and ResUNet-18 sit around 89 % — roughly a five-point gap. The deeper
  stack of 3 × 3 convolutions per resolution level appears to suit thin-line plan
  imagery better than residual blocks do.
- **Neither transfer strategy dominates globally.** Weight copy wins on U-Net
  (+1.28) and VGGUNet-16 (+0.34); distillation wins on ResUNet-18 (+0.37) and
  VGGUNet-19 (+0.49). The effect of the transfer mechanism is an order of magnitude
  smaller than the effect of the encoder choice.
- **Distillation converges earlier** in three of the four pairs (129 vs 141, 140 vs
  146, 100 vs 110), suggesting the teacher signal mostly accelerates convergence
  rather than raising the ceiling.
- **The best configuration overall is VGGUNet-19 with knowledge distillation**:
  95.23 % mIoU and 99.18 % pixel accuracy, reached by epoch 100.

These figures come from the stored outputs of the training notebooks in this
repository and refer to a single run per configuration at seed 9999; they are not
averaged over seeds.

## Repository Structure

```
Heritage-Vision/
├── dataset/
│   ├── dataset.py              # seg_datasets — paired plan/mask loader, class binning
│   ├── labelme.ipynb           # LabelMe JSON → plan/mask JPG rasterisation
│   ├── split_test_train.ipynb  # 80/20 train-test split
│   └── datasets.ipynb          # dataset inspection and RGB mask preview
├── models/
│   ├── unet.py                 # UNET
│   ├── resunet18.py            # RESUNET18
│   ├── resunet50.py            # RESUNET50
│   ├── vgg16.py                # VGGUNET16
│   └── vgg19.py                # VGGUNET19
├── pretraining/
│   ├── pretraining.py          # convnext_base U-Net pre-training script
│   └── pretrain.ipynb          # same as a notebook
├── utils/
│   ├── dataloader_utils.py     # get_loader — train/test DataLoader construction
│   ├── evaluation_utils.py     # calculate_iou, calculate_miou, check_acc
│   ├── save_load_utils.py      # checkpointing with embedded hyperparameters
│   └── visualization_utils.py  # mask colourisation, prediction export, comparison plots
├── unet_training.ipynb         # U-Net: weight copy vs. distillation
├── resunet18_training.ipynb    # ResUNet-18: weight copy vs. distillation
├── vggunet16_training.ipynb    # VGGUNet-16: weight copy vs. distillation
├── vggunet19_training.ipynb    # VGGUNet-19: weight copy vs. distillation
├── requirements.txt            # conda environment export
└── LICENSE                     # MIT
```

Each training notebook follows the same structure, so they can be read in parallel:
hyperparameters → seeding → augmentation → dataset → colour mapping → dataloader →
model → pre-trained loading → weight copy → loss and optimiser → Model-1 training →
Model-2 training → result comparison → prediction export → history graphs.

## Installation

The environment is exported from conda and pinned to Python 3.9.18 / PyTorch 2.1.2
(`osx-64`):

```bash
conda create --name heritage-vision --file requirements.txt
conda activate heritage-vision
```

`requirements.txt` is a conda export, not a pip requirements file — `pip install -r`
will not work on it. On a different platform, install the core dependencies directly:

```bash
conda create -n heritage-vision python=3.9
conda activate heritage-vision
conda install pytorch==2.1.2 torchvision==0.16.2 -c pytorch
pip install albumentations==1.4.0 opencv-python==4.9.0.80 scikit-learn matplotlib tqdm pillow
pip install pretrained-backbones-unet
```

A CUDA-capable GPU is strongly recommended; the notebooks fall back to CPU
automatically via `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.

## Usage

### 1. Prepare the data

Place LabelMe JSON exports in `labelform/`, then run `dataset/labelme.ipynb` to
rasterise them and `dataset/split_test_train.ipynb` to split them. The result is:

```
images/
├── train/   # *Plan.jpg + *Seg.jpg pairs
└── test/
```

### 2. Pre-train the teacher

```bash
python pretraining/pretraining.py
```

This writes `checkpoints/pretrained/pretrained_best.pth.tar`, the weight source and
teacher for every downstream run.

### 3. Train and compare

Open the notebook for the architecture you want and run it top to bottom:

```
unet_training.ipynb        resunet18_training.ipynb
vggunet16_training.ipynb   vggunet19_training.ipynb
```

Each notebook trains both the weight-copy and the distillation variant, prints the
comparison, and writes checkpoints to `checkpoints/<arch>/wc/` and
`checkpoints/<arch>/kd/`.

### 4. Run inference on new plans

```python
from utils.visualization_utils import save_pred_masks

color_mapping = {
    0: (0, 0, 0),        # Walls
    1: (255, 80, 80),    # Iwan
    2: (80, 80, 255),    # Room
    3: (255, 255, 255),  # Background
}

save_pred_masks(
    model=model,
    transform=test_transforms,
    save_folder='./predictions',
    input_folder='./images/test',
    color_mapping=color_mapping,
    hyperparameters=hyperparameters,
)
```

Every `*Plan.jpg` in `input_folder` is written out as a colourised
`*Pred_Mask.jpg`. For a single side-by-side input/ground-truth/prediction figure, use
`plot_segmentation()` from the same module.

## Authors and Mentorship

**Developed by**

- **Ahmet Kemal Cabbar**
- **Enise Nur Yılmaz**

**Under the mentorship of**

- **Sabri Gökmen**
- **Taner Arsan**
- **Figen Özen**

This project was carried out under their joint supervision, bringing together
architectural and heritage-documentation expertise on the one hand and computer
engineering and signal-processing expertise on the other. That pairing shapes the
work directly: the class taxonomy and the decision to treat the iwan as a first-class
category come from the architectural side, while the transfer-learning design that
makes a 131-plan dataset trainable comes from the engineering side.

We are grateful to our mentors for their guidance throughout the project.

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.

Copyright (c) 2024 Ahmet Kemal Cabbar
