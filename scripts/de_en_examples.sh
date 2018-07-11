#!/usr/bin/env bash

# Current best results on de-en_iwslt14 dataset.
# #Parameters: 16948054
# Best BLEU until iter 100000: 32.44
CUDA_VISIBLE_DEVICES=2 nohup python3 child_train.py -T de_en_iwslt_bpe2 -H bpe2_transformer_kt_bias \
    --max-tokens 4000 --log-interval 200 -N net_code_example/transformer_nda.json --restore-file checkpoint_best.pt \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer nag --lr 0.25 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 0.25 --max-update 100000 \
    --extra-options "dec_output_fc=True,attn_linear_bias=False,initializer='kaitao_wn'" \
    > log/de_en_iwslt/transformer_nda_kt_bpe2_bias.log 2>&1 &
