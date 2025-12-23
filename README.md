# Visual-Association based Zero Shot Object Counting

This project reproduces the core components of **VA-Count (ECCV 2024)**:

1. **Exemplar Enhancement Module (EEM)** – Automatically mines *clean single-object exemplars* using **GroundingDINO** proposals and a **CLIP-based binary classifier** to filter noisy or multi-object patches.  
2. **Noise Suppression Module (NSM)** – Trains an exemplar-guided **contrastive density regression** model (on FSC-147) using both positive and negative exemplars to suppress distractors and improve zero-shot counting.

<img width="2441" height="945" alt="image" src="https://github.com/user-attachments/assets/88f30d5a-fc22-4546-ad77-4230e45870d9" />

---

## Why this repository exists

Zero-shot object counting critically depends on the **quality of exemplars**. Noisy crops (multiple objects, clutter, partial instances) lead to poor density estimation. VA-Count addresses this by:

- Automatically generating **high-quality exemplars**
- Explicitly learning to **separate target objects from distractors**
- Using **contrastive supervision** during density estimation

This repository focuses on:
- End-to-end reproduction of the **exemplar generation + filtering pipeline**
- Training a **CLIP + FFN binary classifier** for single-object detection
- Training and evaluating the **Noise Suppression Module**
- Providing **reproducible resources** (datasets, weights, dashboards)

---

## Method overview

### 1. Exemplar Enhancement Module (EEM)

**Goal:** Select top-K *positive* and *negative* exemplars per image.

**Pipeline:**
1. **GroundingDINO** proposes candidate bounding boxes using:
   - Class-specific prompt (e.g., `"eggs, camels"`) → positive pool
   - Generic prompt (e.g., `"object"`) → negative pool
2. **IoU-based filtering** removes negative boxes overlapping heavily with positives.
3. A frozen **CLIP ViT encoder** extracts patch embeddings.
4. A trainable **feed-forward network (FFN)** predicts whether a crop contains a *single object*.
5. Top-K positive and negative exemplars are selected using classifier confidence.

<img width="966" height="570" alt="image" src="https://github.com/user-attachments/assets/2bd61260-175f-4e32-a6a5-6e9118bcd38c" />

Illustration of the single object exemplar filtering with a frozen Clip-vit encoder and a trainable FFN to distinguish single from multiple objects.

---

### 2. Noise Suppression Module (NSM)

**Goal:** Learn robust exemplar-guided density maps.

**Key idea:** Train with both positive and negative guidance.

- **Positive-guided density** → should match ground-truth density.
- **Negative-guided density** → should diverge from ground truth.

<img width="975" height="338" alt="image" src="https://github.com/user-attachments/assets/d987b4b2-5275-4a17-a519-4d3a853002c7" />

Predicted vs. Ground-Truth Density Maps. White boxes show the top 3 exemplars of single object. 

**Training objective:**
Loss = Density Regression Loss + Contrastive Separation Loss

At inference time, only the **positive-guided density map** is integrated to produce the final count.

---

## Repository contents

### Core scripts
- `Project_main_notebook.ipynb` – End-to-end environment setup and pipeline for implementation
- `grounding_pos_dummy.py` – Generate initial positive exemplars to get the main classifier started
- `grounding_pos.py` – Generate class-specific final positive proposals
- `grounding_neg.py` – Generate generic negative proposals
- `datasetmake.py` – Build classifier training dataset
- `biclassify.py` – Train CLIP+FFN single-object classifier
- `FSC_pretrain.py` – NSM pretraining
- `FSC_train.py` – NSM full training
- `FSC_test.py` – Evaluation on FSC-147 splits

- `Project_Report.pdf` – Detailed project report

### Model definitions
- `models_crossvit.py`
- `models_mae_cross.py`
- `models_mae_noct.py`

---
## Data and Resources

### FSC-147 Dataset

- **Images:** - https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view

- **Ground-truth density maps:** - https://archive.org/details/FSC147-GT

---

### Pretrained Weights & Processed Data (Recommended)

A Google Drive bundle containing:
- Preprocessed datasets
- Exemplar crops
- Trained **single-object classifier** and **Noise Suppression Module (NSM)** weights
- Intermediate artifacts required for fast reproduction
- **Download link:** - https://drive.google.com/drive/folders/1jm-lYKTerqOEEg-A0zWRgdlj0paR4vXT

---

### Training Dashboard (Weights & Biases)

- **W&B run dashboard:** - https://api.wandb.ai/links/nitinyadav0497-auburn-university/uvl11odd

This dashboard contains training curves, losses, and intermediate visualisations for the classifier and NSM models.

---
## Acknowledgements: 
I sincerely thank the authors for their open-source implementations and datasets, which enabled reproducibility, learning, and experimentation in zero-shot object counting. If you use this repository or build upon it, please cite the following works:

### VA-Count (ECCV 2024)
```bibtex
@inproceedings{zhu2024vacount,
  title     = {Zero-Shot Object Counting with Good Exemplars},
  author    = {Zhu, H. and Yuan, J. and Yang, Z. and Guo, Y. and Wang, Z. and Zhong, X. and He, S.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```
### FSC-147 / Learning to Count Everything (CVPR 2021)
```bibtex
@inproceedings{ranjan2021lce,
  title     = {Learning to Count Everything},
  author    = {Ranjan, V. and Sharma, U. and Nguyen, T. and Hoai, M.},
  booktitle = {CVPR},
  year      = {2021}
}
```
### GroundingDINO
```bibtex
@article{liu2023groundingdino,
  title   = {Grounding DINO: Marrying DINO with Grounded Pre-training for Open-Set Object Detection},
  author  = {Liu, S. et al.},
  journal = {arXiv preprint arXiv:2303.05499},
  year    = {2023}
}
```
### CLIP
```bibtex
@inproceedings{radford2021clip,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, A. et al.},
  booktitle = {ICML},
  year      = {2021}
}
```




