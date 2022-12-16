## Introduction
This is the PyTorch implementation for **Adaptive Graph Representation Learning for Video Person Re-identification**.

## Get started
```shell script
git clone https://github.com/weleen/AGRL.pytorch /path/to/save
pip install -r requirements.txt
cd torchreid/metrics/rank_cylib && make
```

## Dataset
create dataset directory
```shell script
mkdir data
```
Prepare datasets:
```shell script
├── dukemtmc-vidreid
│   ├── DukeMTMC-VideoReID
│   ├── pose.json
│   ├── split_gallery.json
│   ├── split_query.json
│   └── split_train.json
│
├── ilids-vid
│   ├── i-LIDS-VID
│   ├── pose.json
│   ├── splits.json
│   └── train-test people splits
│
├── mars
│   ├── bbox_test
│   ├── bbox_train
│   ├── info
│   ├── pose.json
│   └── train-test people splits
│
├── prid2011
    ├── pose.json
    ├── prid_2011
    ├── prid_2011.zip
    ├── splits_prid2011.json
    └── train_test_splits_prid.mat
```
`pose.json` is obtained by running [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), we put the files on [Baidu Netdisk](https://pan.baidu.com/s/1RduGEbq-tmfLAHM0k3xa4A) (code: luxr) and 
[Google Driver](https://drive.google.com/drive/folders/1BVEjMava3UQh4jC2bp-tcFo1rOZDB8MS?usp=sharing)(only pose.json, please download the model from Baidu Netdisk).  

More details could be found in [DATASETS.md](DATASETS.md).


## Train
```shell script
bash scripts/train_vidreid_xent_htri_vmgn_mars.sh
```

To use multiple GPUs, you can set `--gpu-devices 0,1,2,3`.

**Note:** To resume training, you can use `--resume path/to/model` to load a checkpoint from which saved model weights and `start_epoch` will be used. Learning rate needs to be initialized carefully. If you just wanna load a pretrained model by discarding layers that do not match in size (e.g. classification layer), use `--load-weights path/to/model` instead.

Please refer to the code for more details.


## Test
create a directory to store model weights `mkdir saved-models/` beforehand. Then, run the following command to test
```shell script
bash scripts/test_vidreid_xent_htri_vmgn_mars.sh
```
All the model weights are available.

## Model

All the results tested with 4 TITAN X GPU and 64GB memory.

| Dataset | Rank-1 | mAP |
| :---: | :---: | :---: |
| iLIDS-VID | 83.7% | - |
| PRID2011  | 93.1% | - | 
| MARS | 89.8% | 81.1% | 
| DukeMTMC-vidreid | 96.7% | 94.2% |


## Citation
Please kindly cite this project in your paper if it is helpful😊:
```
@article{wu2020adaptive,
  title={Adaptive graph representation learning for video person re-identification},
  author={Wu, Yiming and Bourahla, Omar El Farouk and Li, Xi* and Wu, Fei and Tian, Qi and Zhou, Xue},
  journal={IEEE Transactions on Image Processing},
  year={2020},
  publisher={IEEE}
}
```

This project is developed based on [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) and [STE-NVAN](https://github.com/jackie840129/STE-NVAN/).
