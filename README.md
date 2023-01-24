## RelFormer

### Overview
TBA

---

### Setup

1. Download the [Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) dataset and the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset
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
python main.py train /media/yoli/WDC-2.0-TB-Hard-/7Scenes models/backbones/efficient-net-b0.pth datasets/7Scenes/7scenes_training_pairs.csv 7scenes_config.json
```

Testing
```
python main.py test /media/yoli/WDC-2.0-TB-Hard-/7Scenes models/backbones/efficient-net-b0.pth datasets/7Scenes/7scenes_test_pairs/pairs_test_chess.csv 7scenes_config.json --checkpoint_path out/run_24_01_23_10_03_relformer_checkpoint-10.pth
```
