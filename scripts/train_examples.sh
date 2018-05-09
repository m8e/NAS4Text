#!/usr/bin/env bash

# Training the transformer architecture
# [NOTE]: You can modify the "--max-epoch" option to get the best model after 100 epochs.
# [NOTE]: hparams = transformer_share3
# [NOTE]: dropout = 0.1, clip-norm = 2.5
# Result:
#   Epoch   BLEU
#   5       19.33
#   11      23.67
#   20      25.46
#   30      26.24
#   45      27.70
#   70      28.37
#   100     28.75
#   130     28.84
#   160     28.85
#   190     29.08
#   200     29.15
#   215     29.38 (best)
CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 child_train.py \
    -T de_en_iwslt_bpe \
    -H transformer_share3 \
    --max-tokens 1000 \
    --log-interval 500 \
    -N net_code_example/transformer_nd.json \
    --restore-file checkpoint_best.pt \
    --lr-shrink 0.5 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --clip-norm 2.5 \
    --dropout 0.1 \
    --optimizer adam \
    --lr 0.001 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-4 \
    --max-epoch 50 \
    > log/de_en_iwslt/transformer_nd_share3_bpe.log 2>&1 &


# Training the fairseq architecture.
# [NOTE]: You can modify the "--max-epoch" option to get the best model after 100 epochs.
# [NOTE]: hparams = fairseq_share3
# [NOTE]: dropout = 0.2, clip-norm = 0.1
# Result:
#   Epoch   BLEU
#   5       18.85
#   10      24.07
#   15      25.39
#   20      26.75
#   30      27.56
#   40      28.25
#   50      27.98
#   70      28.57
#   90      28.63
#   100     28.86 (best)
CUDA_VISIBLE_DEVICES=0,1,2 nohup python3 child_train.py \
    -T de_en_iwslt_bpe \
    -H fairseq_share3 \
    --max-tokens 1000 \
    --log-interval 500 \
    -N net_code_example/fairseq.json \
    --restore-file checkpoint_best.pt \
    --lr-shrink 0.5 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --clip-norm 0.1 \
    --dropout 0.2 \
    --optimizer adam \
    --lr 0.001 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-4 \
    > log/de_en_iwslt/fairseq_share3_bpe.log 2>&1 &
