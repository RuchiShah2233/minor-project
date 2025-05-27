# Robust CNN Architecture for Contact-Free Vital-Signs Monitoring  
_Remote Photoplethysmography (rPPG)_  

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red) ![License: MIT](https://img.shields.io/badge/License-MIT-green)

## ✨ Overview  
This project implements a **convolutional-neural-network (CNN) pipeline** that extracts remote photoplethysmography (rPPG) signals from ordinary RGB video and predicts **heart-rate (HR), respiratory-rate (RR) and SpO₂** under challenging motion and illumination.  
Key stages include robust ROI selection with **MediaPipe Face Mesh**, chrominance-based signal extraction (**CHROM**), multi-scale spectral featurization (DFT, CWT, MS-LTSS) and a custom spatial-temporal CNN trained on five public rPPG datasets. The codebase contains Jupyter notebooks for data preparation and training plus lightweight Python scripts for real-time inference and benchmarking.

## 🗂️ Repository Structure
```text
├── DataPrep.ipynb              # build datasets + featurize signals  
├── FeatureProcessing/          # classic & custom feature maps  
├── Cnn_Train.ipynb             # model architecture & training loop  
├── test.py | testBHrppg.py     # quick evaluation scripts  
├── images/                     # sample frames & result plots  
└── README.md
```

## 📊 Datasets  
| Dataset | Subjects / Videos | Highlights | Licence |
|---------|-------------------|------------|---------|
| **BH-rPPG** | 12 subjects, 3 lighting setups | strong illumination shifts | [Link](https://github.com/ubicomplab/bh-rppg-dataset) |
| **UBFC-rPPG / UBFC-Phys** | 42 subjects | baseline & multi-parameter ground-truth | [Link](https://sites.google.com/view/ubfc-rppg) |
| **MAHNOB-HCI** | 30 subjects, multimodal | emotion-eliciting protocol | [Link](https://mahnob-db.eu/hci-tagging/) |
| **COHFACE** | 40 subjects, 160 videos | synchronised HR & RR | [Link](https://www.idiap.ch/en/dataset/cohface) |
| **PURE** | 10 subjects, head-motion scenarios | high-resolution, varied motion | [Link](https://www.lpb.rwth-aachen.de/pure-dataset) |

> **Note:** Download instructions are provided in the notebook `DataPrep.ipynb`.

## 🏗️ Pipeline
1. **ROI Detection** – Face landmarks & mask via *MediaPipe Face Landmarker* ([MediaPipe](https://developers.google.com/mediapipe/solutions/vision/face_landmarker))  
2. **Signal Extraction** – CHROM algorithm for chrominance pulse traces ([de Haan & Jeanne, 2013](https://ieeexplore.ieee.org/document/6514913))  
3. **Denoising & Normalisation** – Band-pass filter (0.7–4 Hz) + detrending  
4. **Feature Engineering** –  
   - Discrete Fourier Transform (DFT)  
   - Continuous Wavelet Transform (CWT)  
   - Multi-Scale Long-Term Statistical Spectrum (MS-LTSS) ([Wang et al., 2021](https://arxiv.org/abs/2102.05011))  
5. **CNN Model** – Spatial-temporal backbone with residual blocks ([TensorFlow](https://www.tensorflow.org/) / [PyTorch](https://pytorch.org/))  
6. **Training & Evaluation** – Cross-dataset protocol + rPPG-Toolbox utilities ([rPPG-Toolbox](https://github.com/phuselab/rPPG-Toolbox))
7. **Deployment** – `test.py` streams webcam frames and outputs live HR/RR/SpO₂.

## ⚙️ Installation
```bash
git clone https://github.com/RuchiShah2233/minor-project.git
cd minor-project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # generates from notebooks
```

## 🚀 Quick Start
```bash
1. Prepare datasets & features
jupyter nbconvert --to notebook --execute DataPrep.ipynb

2. Train the network
jupyter nbconvert --to notebook --execute Cnn_Train.ipynb

3. Run real-time demo
python test.py --source webcam --model checkpoints/model_best.pth
```

## 📈 Results

| Metric (MAE)   | HR (bpm) | RR (breaths/min) | SpO₂ (%) |
|----------------|----------|------------------|----------|
| **Validation*** | 1.92     | 1.45             | 1.12     |

\* Five-fold cross-validation averaged over BH-rPPG lighting conditions.

## 🛠️ Tools & Libraries
- Python 3.10, Jupyter Lab
- TensorFlow 2 / tf-keras & PyTorch 2
- NumPy / SciPy for DSP
- OpenCV for video I/O
- MediaPipe for face mesh extraction
- Google for Developers
- Matplotlib for visualisation
- rPPG-Toolbox utilities for evaluation 

## 📜 Citation & References
If you use this code, please cite the original algorithm and dataset papers:
- [de Haan & Jeanne, “Robust Pulse Rate from Chrominance-Based R-PPG” (CHROM)](https://pubmed.ncbi.nlm.nih.gov/23744659/?utm_source=chatgpt.com)
- [Wang et al. “Algorithmic Principles of Remote-PPG”](https://pure.tue.nl/ws/files/31563684/TBME_00467_2016_R1_preprint.pdf?utm_source=chatgpt.com)
- [Comprehensive review of rPPG methods (2023)](https://www.medrxiv.org/content/10.1101/2023.10.12.23296882.full?utm_source=chatgpt.com)


## 📄 License
This project is released under the MIT License – see LICENSE.

