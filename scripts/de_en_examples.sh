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

# Generate script.
BEAM=5; ALPHA=1.1; VARS="49 59"; CUDA_VISIBLE_DEVICES=3; for i in ${VARS}
do
    python3 child_gen.py -T de_en_iwslt_bpe2 -H bpe2_transformer_kt_bias -N net_code_example/transformer_nda.json \
    --path checkpoint${i}.pt --output-file output_pt${i}_beam${BEAM}.txt --use-task-maxlen --beam ${BEAM} \
    --lenpen ${ALPHA}
done; for i in ${VARS}
do
    python3 score.py -r data/de_en_iwslt/test.iwslt.de-en.en \
    -s translated/de_en_iwslt_bpe2/bpe2_transformer_kt_bias/transformer_nda/output_pt${i}_beam${BEAM}.txt
done

# Perl script example.
perl scripts/multi-bleu.perl data/de_en_iwslt/test.iwslt.de-en.en < translated/de_en_iwslt_bpe2/bpe2_transformer_kt_bias/block_s_example/output_pt2_beam5.txt
