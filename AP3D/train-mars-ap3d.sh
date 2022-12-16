python train.py --root ../../mevid -d mars --arch ap3dres50 --gpu 2,3 --save_dir log-mars-ap3d #
python test-all.py --root ../../mevid/ -d mars --arch ap3dres50 --gpu 0 --resume log-mars-ap3d
