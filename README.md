## Beyond Familiar Landscapes: Exploring the Limits of Relative Pose Regressors in New Environments
Official PyTorch implementation of our paper: [Beyond Familiar Landscapes: Exploring the Limits of Relative Pose Regressors in New Environments](https://www.sciencedirect.com/science/article/pii/S1077314225003522?via%3Dihub)

**Relformer** is a hybrid CNN-Transformer architecture designed to push the boundaries of Relative Pose Regression (RPR). By combining a convolutional backbone with transformer encoders and hypernetwork-driven adaptation, Relformer achieves superior zero-shot localization and adaptability in unseen environments.

---

## 🚀 Key Features

* **Hybrid CNN-Transformer Architecture**: Leverages EfficientNet for feature extraction and Transformers for global context modeling.
* **Hypernetwork Adaptation**: Dynamically refines pose estimates using auxiliary MLP heads learned via hypernetworks, significantly improving performance in new scenes.
* **Dual-Branch Design**: Decoupled branches for translation and rotation for specialized feature processing.
* **Geometric Pose Loss**: Optimized end-to-end with a combination of standard pose loss and a novel geometric loss.
* **Multi-Scene Support**: Ready-to-use configurations for standard benchmarks like 7Scenes and Cambridge Landmarks.

---

## 🛠️ Architecture Overview

The Relformer architecture processes image pairs through a shared backbone, followed by Transformer encoders that aggregate paired feature maps into latent representations. The final pose is regressed using a main branch and an adaptive residual branch powered by hypernetworks.

![Improving the Zero-Shot Localization of Relative Pose Regressors](./img/teaser.jpg)
---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yolish/relformer.git
cd relformer

```

### 2. Set Up Environment

We recommend using a Conda environment:

```bash
conda create -n relformer python=3.8
conda activate relformer
pip install -r requirements.txt

```

### 3. Prepare Datasets

Download the following datasets and place them in the `datasets/` directory:

* [7Scenes Dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
* [Cambridge Landmarks Dataset](https://www.repository.cam.ac.uk/handle/1810/251342)

---

## 🚦 Quick Start

### Training

To train the Relformer model on the 7Scenes dataset:

```bash
python main.py --mode train \
    --dataset_path ./datasets/7Scenes \
    --rpr_backbone_path models/backbones/efficient-net-b0.pth \
    --labels_file datasets/7Scenes/7scenes_training_pairs.csv \
    --config_file config/7scenes_config.json \
    --gpu 0

```

### Testing

To evaluate a trained checkpoint:

```bash
python main.py --mode test \
    --dataset_path ./datasets/7Scenes \
    --test_labels_file datasets/7scenes_test_NN/NN_7scenes_chess.csv \
    --checkpoint_path checkpoints/relformer_all.pth \
    --config_file config/7scenes_config.json \
    --gpu 0

```

---

## 🧠 Pretrained Models

| Model | Training Set | Download |
| --- | --- | --- |
| `relformer_all.pth` | All 7Scenes | [Download Here](https://www.google.com/search?q=https://example.com/models/relformer_all.pth) |
| `relformer_nofire.pth` | 7Scenes (excluding "Fire") | [Download Here](https://www.google.com/search?q=https://example.com/models/relformer_nofire.pth) |

---

## ⚙️ Configuration Guide

The repository includes several configuration presets in the `config/` directory:

* `7scenes_config.json`: **Full Model** (Transformer, 6D Rotation, Geometric Loss, Hypernetworks).
* `7scenes_config_deltanet_baseline.json`: Standard RPR baseline using Quaternions.
* `7scenes_config_deltanet_transformer_6d.json`: Transformer-based matching with 6D rotation representation.

---

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{idan2026beyond,
  title={Beyond familiar landscapes: Exploring the limits of relative pose regressors in new environments},
  author={Idan, Ofer and Shavit, Yoli and Keller, Yosi},
  journal={Computer Vision and Image Understanding},
  pages={104629},
  year={2026},
  publisher={Elsevier}
}

```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.



