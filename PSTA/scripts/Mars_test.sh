#!/bin/scripts

python Test.py  --arch 'PSTA'\
                --dataset 'mars'\
                --test_sampler 'Begin_interval'\
                --triplet_distance 'cosine'\
                --test_distance 'cosine'\
                --test_path 'log/PSTA_best_model.pth'
