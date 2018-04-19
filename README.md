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

## Train Child Model

```bash
python train.py [More options]

# Example on de-en iwslt dataset
python child_train.py \
    -T de_en_iwslt \
    -H normal \
    --max-tokens 500 \
    --log-interval 10 \
    -N net_code_example/default.json
```

## Model Storage

```
models/
    <task-name>/
        <hparams-set-name>/
            <net-code-filename-without-ext>/
                checkpoint1.pt
                checkpoint_best.pt
                checkpoint_last.pt
                ...
```

## Inference Child Model

TODO

## Train Teacher Model

TODO
