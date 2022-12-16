#!/bin/bash
i=0
while((i<10))
do
python train_vidreid_xent_htri.py -d prid2011 \
                                  --seq-len 8 \
                                  --train-batch 16 \
                                  --test-batch 16 \
                                  --num-instances 4 \
                                  --train-sample restricted \
                                  --test-sample evenly \
                                  --train-sampler RandomIdentitySamplerV1 \
                                  --optim adam \
                                  --soft-margin \
                                  --max-epoch 400 \
                                  --lr 1e-4 \
                                  --stepsize 100 200 300 \
                                  -a vmgn \
                                  --num-split 4 \
                                  --pyramid-part \
                                  --num-gb 2 \
                                  --use-pose \
                                  --learn-graph \
                                  --flip-aug \
                                  --print-last \
                                  --gpu-devices 0 \
                                  --eval-step 1 \
                                  --dist-metric cosine \
                                  --split-id $i \
                                  --consistent-loss \
                                  --save-dir log/video/vmgn/prid2011-ngb2-consistent/split"$i"
let i=$i+1
done
