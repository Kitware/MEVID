#!/bin/bash
python train_vidreid_xent_htri.py -d mars \
                                  -a vmgn \
                                  --evaluate \
                                  --seq-len 8 \
                                  --test-sample evenly \
                                  --num-split 4 \
                                  --pyramid-part \
                                  --num-gb 2 \
                                  --learn-graph \
                                  --gpu-devices 0 \
                                  --dist-metric cosine \
                                  --load-weights log/vmgn/mars-ngb2-consistent/checkpoint_ep80.pth.tar \
                                  --save-dir log/mars
