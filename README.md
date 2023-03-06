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

---

### Pretrained models:
You can download our pretrained models for the 7Scenes dataset (trained over all scenes / without fire scene), from here: [pretrained models](https://drive.google.com/file/d/1MyfS6a_05u2KFVIaUoLYSkuF-i4jTgdo/view?usp=sharing)

1. relformer_DeltanetEnc_6d_all.pth: model trained with config/7scenes_config_deltanet_transformer_encoder_6d.json over 7Scenes dataset including all scenes
2. relformer_DeltanetEnc_6d_nofire.pth: model trained with config/7scenes_config_deltanet_transformer_encoder_6d.json over 7Scenes dataset over 6 scenes while 'fire' scene kept out of training

### Usage
Training 
```
python main.py --mode train --dataset_path /media/yoli/WDC-2.0-TB-Hard-/7Scenes --rpr_backbone_path models/backbones/efficient-net-b0.pth --labels_file datasets/7Scenes/7scenes_training_pairs.csv --config_file config/7scenes_config_deltanet_transformer_encoder_6d.json --gpu 0
```
Testing
```
python main.py --mode test --dataset_path /media/yoli/WDC-2.0-TB-Hard-/7Scenes --rpr_backbone_path models/backbones/efficient-net-b0.pth --test_labels_file datasets/7Scenes/7scenes_test_pairs/pairs_test_chess.csv --config_file config/7scenes_config_deltanet_transformer_encoder_6d.json --checkpoint_path checkpoints/relformer_DeltanetEnc_6d_all.pth --gpu 0
```


### Configurations  (under './config'):

7scenes_config_deltanet_baseline.json: no feature matching between query and reference images, orientation representation is quaternion

7scenes_config_deltanet_conv.json: feature matching between query and reference images is convolution, orientation representation is quaternion

7scenes_config_deltanet_transformer_encoder: feature matching between query and reference images is transformer encoder, orientation representation is quaternion

7scenes_config_deltanet_transformer_encoder_6d: feature matching between query and reference images is transformer encoder, orientation representation is 6d 

7scenes_config_deltanet_transformer_encoder_9d: feature matching between query and reference images is transformer encoder, orientation representation is 9d 

7scenes_config_deltanet_transformer_encoder_10d: feature matching between query and reference images is transformer encoder, orientation representation is 10d 
 