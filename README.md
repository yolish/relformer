## Learning to Localize in Unseen Scenes with Relative Pose Regressors
Official PyTorch implementation of Learning to Localize in Unseen Scenes with Relative Pose Regressors, for details see our paper [Learning to Localize in Unseen Scenes with Relative Pose Regressors]

The figure below illustrates our approach: a query image and reference image are put to our architecture, we then extract informative features using a convolutional backbone, 
concatenate the query and reference images features and use two relformers blocks separately attend to position-  and orientation- . 
![Learning to Localize in Unseen Scenes with Relative Pose Regressors Illustration](./img/teaser.jpg)

---

### Repository Overview 

This code implements:

1. Training of a multiple architectures for multi-scene relative pose regression 
2. Testing code

---

### Prerequisites

In order to run this repository you will need:

1. Python3 (tested with Python 3.7.7, 3.8.5), PyTorch
2. Set up dependencies with ```pip install -r requirements.txt```
3. Download the [Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) dataset and the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset

1. Setup a conda env:
```
conda create -n loc python=3.7
pip install torch==1.4.0 torchvision==0.5.0
pip install scikit-image
pip install efficientnet-pytorch
pip install pandas
pip install transforms3d
conda activate loc
```

---

### Usage and Pretrained Models 
Training 
```
python main.py --mode train --dataset /media/yoli/WDC-2.0-TB-Hard-/7Scenes --rpr_backbone_path models/backbones/efficient-net-b0.pth --labels_file datasets/7Scenes/7scenes_training_pairs.csv --config_file 7scenes_config_deltanet_transformer_encoder_10d.json --gpu 2
```
Testing
```
python main.py --mode test --dataset /media/yoli/WDC-2.0-TB-Hard-/7Scenes --rpr_backbone_path models/backbones/efficient-net-b0.pth --test_labels_file datasets/7Scenes/7scenes_test_pairs/pairs_test_chess.csv --config_file 7scenes_config.json --checkpoint_path out/run_24_01_23_10_03_relformer_checkpoint-10.pth
```
