[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-person-re-id-fantastic-techniques-and/person-re-identification-on-mars)](https://paperswithcode.com/sota/person-re-identification-on-mars?p=video-person-re-id-fantastic-techniques-and)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-person-re-id-fantastic-techniques-and/person-re-identification-on-prid2011)](https://paperswithcode.com/sota/person-re-identification-on-prid2011?p=video-person-re-id-fantastic-techniques-and)

# Video Person Re-ID: Fantastic Techniques and Where to Find Them
Official Pytorch Implementation of the paper: [Paper](https://arxiv.org/pdf/1912.05295.pdf)   
**Video Person Re-ID: Fantastic Techniques and Where to Find Them** *(Accepted in AAAI'20)*  
*Priyank Pathak,  Amir Erfan Eshratifar,  Michael Gormish*   
*(Work done in collaboration of NYU and Clarifai)*    
If you find this useful, feel free to cite this
```
@misc{pathak2019video,
    title={Video Person Re-ID: Fantastic Techniques and Where to Find Them},
    author={Priyank Pathak and Amir Erfan Eshratifar and Michael Gormish},
    year={2019},
    eprint={1912.05295},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

Extension of Work done for Videos 
* [Revisiting Temporal Modeling for Video-based Person ReID](https://github.com/jiyanggao/Video-Person-ReID) || [paper](https://arxiv.org/pdf/1805.02104.pdf)
* [Bag of Tricks and A Strong ReID Baseline](https://github.com/michuanhaohao/reid-strong-baseline) || [paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf) 
* [Online-Soft-Mining-and-Class-Aware-Attention](https://github.com/ppriyank/-Online-Soft-Mining-and-Class-Aware-Attention-Pytorch) || [paper](https://arxiv.org/pdf/1811.01459v2.pdf)

## CORRECTION 
PRID results have been reported for split 0. 
Please consider the current SOTA as R-1 : 95.5 R-5: 100 R10: 100
for **Global-local temporal representations for video person re-identification. In ICCV, 2019**

## Specifications  
Pytorch == 1.3.0  
Python3   

## MODEL

<img src="https://github.com/ppriyank/Video-Person-Re-ID-Fantastic-Techniques-and-Where-to-Find-Them/blob/master/images/diag.png" width="900">


## RESULTS




### LOSS

```
loss = ide_loss + (1 - beta_ratio)*triplet_loss + center_loss * cetner_loss_weight + beta_ratio * osm_caa_loss + attention_loss
```

### Performance

   
Use `59` configuration in `cl_centers.conf` and `vals.conf` for **MARS DATASET** and `27` and `24` configuration in `prid.conf` for **prid DATASET**


**MARS DATASET** 

| Model            | mAP (re-rank)  | CMC-1 (re-rank) | CMC-5 (re-rank)| CMC-20 (re-rank)|
| :--------------- | -------------: | -------------: | -------------: | ------------: | 
| SOTA (w/o re-rank) (Fu et al.)[1]     |   81.2 (-)  | 86.2 (-) | 95.7 (-) |  - (-)   |
| SOTA (with re-rank) (Fu et al.)[1]   |   80.8 (87.7)  | 86.3(87.2) | 95.7(96.2) | 98.1(98.6) |
| Baseline     |  76.7 (84.5) | 83.3 (85.0) | 93.8 (94.7) | 97.4 (97.7)|
| Baseline + BOT    |    81.3 (88.4) | 87.1 (87.6) | 95.9 (96.0) | 98.2 (98.4) |
|  Baseline + BOT + OSM Loss    | 82.4 (88.1) |  87.9 (87.6) | 96.0 (95.7)  | 98.0 (98.5)  |
| (Proposed) Baseline + BOT + OSM Loss + CL Centers    |  81.2 **(88.5)**  | 86.3 **(88.0)**  | 95.6 (96.1)  | **98.2** (98.5) |
| (Proposed) B-BOT + Attn-CL loss    | **82.9**(87.8)   | **88.6**(88.0)  | 96.2(95.4)  | 98.0(98.3) |


**PRID DATASET**   

| Model            | CMC-1 | CMC-5 | CMC-20 |
| :--------------- | ----------: | ----------: | ----------: |
| SOTA  (Zeng, Tian, and Wu)[2]     |  96.1%  | 99.5  | -  |  
| Baseline + BOT + OSM Loss + CL Centers    |  93.1  |  98.9  | 100.0 |
| Baseline + BOT + OSM Loss + CL Centers (pretrained on MARS)   |  96.6  | 100  | 100  | 
 


## DATASET

MARS dataset: 

<img src="https://github.com/ppriyank/Video-Person-Re-ID-Fantastic-Techniques-and-Where-to-Find-Them/blob/master/images/data.jpg" width="400">


## MARS Dataset configuration (59th)

### OSM LOSS hyper paramters
* sigma =  0.9047814732165316
* alpha =  2.8436551583293728
* l =  0.5873389293193368

### Triplet loss

* margin =  4.4132437486402204e-05

### (1 - beta_ratio) x Triplet_Loss + (beta_ratio) x OSM_Loss  
* beta_ratio =  1.0 (only osm loss used)

### Optimizaer hyperparameters
* gamma =  0.3282654557691594
* weight_decay = 0.0005


## NOTE 
Change the dataset path in `tools/data_manager.py` line 21 : `storage_dir = "/beegfs/pp1953/"`    
(apologies for the inconvenience)

## bag of tricks   
normal architecrture : `args.arch = "ResNet50ta_bt"`    
For attention loss :  `args.arch = "ResNet50ta_bt2"`    

`python bagoftricks.py --name="_CL_CENTERS_" --validation-training --cl-centers --opt=8`  
`python bagoftricks.py --name="_triplet_OSM_only_" --validation-training --use-OSMCAA --opt=8`  
`python bagoftricks.py --name="_triplet_only_" --validation-training --opt=8`   
`python bagoftricks.py --name="_ilidsvid_" --validation-training --opt=8`   
`python bagoftricks.py --name="_prid_" --validation-training --opt=24`   
`python config_trainer.py --focus=map --dataset=mars --opt=59 --name=_mars_attncl_centers_ --cl-centers --attn-loss`

## hyper parameter optimization   

`python hyper_supervise_validation.py --focus="map" --opt=8`       
`python hyper_supervise_validation.py --focus="rerank_map" --opt=8`      
`python hyper_supervise_validation.py --focus="map" --sampling="inteliigi" --opt=8`      
`python hyper_supervise_validation.py --focus="rerank_map" --sampling="inteliigi" --opt=8`    

python extract_query_feature.py --name="_CL_CENTERS_" --cl-centers

Ref: 

[1] STA:  Spatial-Temporal Attention for Large-Scale Video-based Person Re-Identification  
[2] Person Re-identification with Hierarchical Deep Learning Feature and efficient XQDA Metric
