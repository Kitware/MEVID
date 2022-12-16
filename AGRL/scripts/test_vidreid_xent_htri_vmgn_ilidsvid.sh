#!/bin/bash
i=0
while((i<10))
do
python train_vidreid_xent_htri.py -d ilidsvid \
                                  --evaluate \
                                  --seq-len 8 \
                                  --test-batch 16 \
                                  --test-sample evenly \
                                  -a vmgn \
                                  --num-split 4 \
                                  --pyramid-part \
                                  --num-gb 2 \
                                  --use-pose \
                                  --learn-graph \
                                  --gpu-devices 0 \
                                  --dist-metric cosine \
                                  --split-id $i \
                                  --load-weights saved-models/ilidsvid/split"$i"/model_ilidsvid.pth.tar \
                                  --save-dir log/ilidsvid/split"$i"
let i=$i+1
done
