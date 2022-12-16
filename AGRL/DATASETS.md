## How to prepare data

Create a directory to store reid datasets under this repo via
```bash
cd AGRL.pytorch/
mkdir data/
```

If you wanna store datasets in another directory, you need to specify `--root path_to_your/data` when running the training code. Please follow the instructions below to prepare each dataset. After that, you can simply do `-d the_dataset` when running the training code. 

Please do not call image dataset when running video reid scripts, otherwise error would occur, and vice versa.

### Video ReID

**MARS** [8]:
1. Create a directory named `mars/` under `data/`.
2. Download dataset to `data/mars/` from http://www.liangzheng.com.cn/Project/project_mars.html.
3. Extract `bbox_train.zip` and `bbox_test.zip`.
4. Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put `info/` in `data/mars` (we want to follow the standard split in [8]). The data structure would look like:
```
mars/
    bbox_test/
    bbox_train/
    info/
```
5. Use `-d mars` when running the training code.

**iLIDS-VID** [11]:
1. The code supports automatic download and formatting. Simple use `-d ilidsvid` when running the training code. The data structure would look like:
```
ilids-vid/
    i-LIDS-VID/
    train-test people splits/
    splits.json
```

**PRID** [12]:
1. Under `data/`, do `mkdir prid2011` to create a directory.
2. Download dataset from https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/ and extract it under `data/prid2011`.
3. Download the split created by [iLIDS-VID](http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html) from [here](http://www.eecs.qmul.ac.uk/~kz303/deep-person-reid/datasets/prid2011/splits_prid2011.json), and put it in `data/prid2011/`. We follow [11] and use 178 persons whose sequences are more than a threshold so that results on this dataset can be fairly compared with other approaches. The data structure would look like:
```
prid2011/
    splits_prid2011.json
    prid_2011/
        multi_shot/
        single_shot/
        readme.txt
```
4. Use `-d prid2011` when running the training code.

**DukeMTMC-VideoReID** [16, 23]:
1. Use `-d dukemtmcvidreid` directly.
2. If you wanna download the dataset manually, get `DukeMTMC-VideoReID.zip` from https://github.com/Yu-Wu/DukeMTMC-VideoReID. Unzip the file to `data/dukemtmc-vidreid`. Ultimately, you need to have
```
dukemtmc-vidreid/
    DukeMTMC-VideoReID/
        train/ # essential
        query/ # essential
        gallery/ # essential
        ... (and license files)
```


## Dataset loaders
These are implemented in `dataset_loader.py` where we have two main classes that subclass [torch.utils.data.Dataset](http://pytorch.org/docs/master/_modules/torch/utils/data/dataset.html#Dataset):
* [VideoDataset](https://github.com/KaiyangZhou/deep-person-reid/blob/master/dataset_loader.py#L38): processes video-based person reid datasets.

These two classes are used for [torch.utils.data.DataLoader](http://pytorch.org/docs/master/_modules/torch/utils/data/dataloader.html#DataLoader) that can provide batched data. Data loader with `VideoDataset` outputs batch data of `(batch, sequence, channel, height, width)`.