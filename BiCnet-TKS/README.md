## [BiCnet-TKS: Learning Efficient Spatial-Temporal Representation for Video Person Re-Identification](https://arxiv.org/abs/2104.14783)

#### Requirements: Python=3.8 and Pytorch=1.7.0

### Training and test

```Shell
# For MARS
python train.py -d mars --root ../datasets/MARS/ --save_dir log-mars #
python test.py -d mars --root ../datasets/MARS/ --save_dir log-mars #
  ```
  

### Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{BiCnet-TKS,
  title={BiCnet-TKS: Learning Efficient Spatial-Temporal Representation for Video Person Re-Identification},
  author={Ruibing Hou and Hong Chang and Bingpeng Ma and  Rui Huang and Shiguang Shan},
  booktitle={CVPR},
  year={2021}
}
```
