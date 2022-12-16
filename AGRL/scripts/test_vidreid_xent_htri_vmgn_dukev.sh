#!/bin/bash
python train_vidreid_xent_htri.py -d dukemtmcvidreid \
                                  -a vmgn \
                                  --evaluate \
                                  --seq-len 8 \
                                  --test-sample evenly \
                                  --num-split 4 \
                                  --pyramid-part \
                                  --num-gb 2 \
                                  --use-pose \
                                  --learn-graph \
                                  --gpu-devices 0 \
                                  --dist-metric cosine \
                                  --load-weights saved-models/dukemtmc-vidreid/model_dukev.pth.tar \
                                  --save-dir log/dukev
