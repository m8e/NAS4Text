# NAS4Text
Neural Architecture Search on Text Tasks.

## Requirements

- Python 3
- Pytorch
- tqdm (optional)

## Data Format

See docstring of [`libs/utils/data_processing.py`](libs/utils/data_processing.py).

## Net Code Format

See docstring of [`libs/layers/net_code.py`](libs/layers/net_code.py).

User can put their net code files in `usr_net_code/`, which is ignored by git.

In training and inference, use `-N` to specify the net code file.

## Search Space Definition

See search spaces of [`libs/utils/search_space.py`](libs/utils/search_space.py).

## Train Child Model

```bash
python child_train.py [More options]

# Example on de-en iwslt dataset
python child_train.py \
    -T de_en_iwslt \
    -H normal \
    --max-tokens 500 \
    --log-interval 10 \
    -N net_code_example/default.json
```

More examples can be seen in [`scripts/train_examples.sh`](scripts/train_examples.sh)

## Model Storage

```
models/
    <task-name>/
        <hparams-set-name>/
            <experiment-name>/
                <net-code-filename-without-ext>/
                    checkpoint1.pt
                    checkpoint_best.pt
                    checkpoint_last.pt
                    ...
```

## Inference Child Model

```bash
python child_gen.py [More options]

# Example on de-en iwslt dataset
python child_gen.py \
    -T de_en_iwslt \
    -H normal \
    -N net_code_example/default.json \
    --max-tokens 500 \
    --path checkpoint_last.pt \
    --use-task-maxlen \
    --output-file output_pt1.txt
```

Then compute BLEU:

```bash
perl scripts/multi-bleu.perl data/de_en_iwslt/test.iwslt.de-en.en < translated/[output file path above]
```

## NAO Training

```bash
python nao_train.py [More options]

# Example on de-en iwslt dataset
# TODO
```
