# STMN
This repository contains a [Pytorch](https://pytorch.org/) implementation for our [STMN (ICCV 2021)](https://cvlab-yonsei.github.io/projects/STMN/). Our code is released only for scientific or personal use. Please contact us for commercial use.

## Requirements

- Python 3.6.8
- Pytorch 1.1.0
- Cuda 10.0
- Cudnn 7.5
- Pillow
- progressbar2
- tqdm
- pandas

## Getting Started

### Datasets
- Create your own database files, MARS for example, as follows:
```
cd database
python create_MARS_database.py \
    --data_dir 'path/to/MARS/' \
    --info_dir /path/to/MARS dataset/MARS-evaluation/info/ \
    --output_dir ./MARS_database/
```

### Train
- You can train our model using the below commands. Note that, in advance, you have to change variables 
'TRAIN_TXT', 'TRAIN_INFO', 'TEST_TXT', 'TEST_INFO', and 'QUERY_INFO' in train.sh 
according to which dataset you want to use for the triaining.
```
cd smem_tmem
sh train.sh
```

### Test
- You can test a pre-trained model using the below commands. Similarly, you have to change variables 
'TRAIN_TXT', 'TRAIN_INFO', 'TEST_TXT', 'TEST_INFO', and 'QUERY_INFO' in test.sh 
according to which dataset you want to use for evaluation.
- Specify the path to pre-trained model parameters using 'LOAD_CKPT'
```
cd smem_tmem
sh test.sh
```

<!--
## Citation
Please cite our paper if you find the code useful for your research.
```
@inproceedings{eom2021learning,
  title={Learning Disentangled Representation for Robust Person Re-identification},
  author={Eom, Chanho and Ham, Bumsub},
  booktitle={Advances in Neural Information Processing Systems},
  pages={5298--5309},
  year={2019}
}
```
-->

## Acknowledgements
Our code is inspired by [STE-NVAN](https://github.com/jackie840129/STE-NVAN)
